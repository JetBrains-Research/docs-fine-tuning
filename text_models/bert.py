import os.path
import tempfile
from typing import List, Union, Callable, Optional

import numpy as np
import wandb
from omegaconf import DictConfig, ListConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from transformers import BertConfig, BertModel, AutoTokenizer
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from data_processing.util import Section, Corpus
from text_models import AbstractModel, TrainTypes
from text_models.dapt_tasks import AbstractPreTrainingTask
from text_models.dapt_tasks import dapt_tasks
from text_models.evaluation import ValMetric
from text_models.task_models import AbstractTask


class BertDomainModel(AbstractModel):
    """
    Text Transformer-Based model, that can be fine-tuned on docs through various tasks
    and can also be used to map sentences / text to embeddings.

    :param domain_adaptation_tasks: List of fine-tuning tasks that will be used for fine-tuning on docs
    :param cnf_dapt_tasks: Configuration for fine-tuning tasks
    :param pretrained_model: The name of pretrained text model
    :param start_train_from: Can be: bugs/task/None
    :param seed: Random seed
    :param save_to_path: Where the trained model should be saved
    """

    def __init__(
        self,
        target_task: AbstractTask = None,
        cnf_dapt_tasks: Union[DictConfig, ListConfig] = None,
        domain_adaptation_tasks: List[str] = None,
        pretrained_model: str = "bert-base-uncased",
        start_train_from: Optional[str] = None,  # 'bugs'/'task'/None
        seed: int = 42,
        save_to_path: str = "./",
        report_wandb: bool = False,
        wandb_config: Union[DictConfig, ListConfig] = None,
        hp_search_mode: bool = False,
    ):
        super().__init__(pretrained_model=pretrained_model, seed=seed, save_to_path=save_to_path)

        if target_task is not None:
            self.task = target_task
            self.tapt_data = self.task.tapt_data
            self.epochs = self.task.config.epochs

        if domain_adaptation_tasks is None:
            domain_adaptation_tasks = ["mlm"]

        self.start_train_from = start_train_from
        self.report_wandb = report_wandb
        self.wandb_config = wandb_config
        self.hp_search_mode = hp_search_mode

        self.vocab_size = None
        self.tokenizer = None
        self.best_metric = 0.0

        if cnf_dapt_tasks is not None:
            self.domain_adaptation_tasks = [
                dapt_tasks[name](**cnf_dapt_tasks[name]) for name in domain_adaptation_tasks  # type: ignore
            ]

    name = "BERT"

    def train_task(self, corpus: Section):
        dumb_model_name = self.__create_and_save_model_from_scratch()

        word_embedding_model = models.Transformer(dumb_model_name, max_seq_length=self.task.config.max_len)
        self.model = self.task.train(
            word_embedding_model,
            os.path.join(self.save_to_path, self.name + "_" + TrainTypes.TASK),
            "task/global_steps",
            self.report_wandb,
            self.hp_search_mode,
        )
        if self.hp_search_mode:
            self.best_metric = self.model.best_score  # type: ignore

    def train_pt_task(self, corpus: Section):
        word_embedding_model = models.Transformer(self.pretrained_model, max_seq_length=self.task.config.max_len)
        self.model = self.task.train(
            word_embedding_model,
            os.path.join(self.save_to_path, self.name + "_" + TrainTypes.PT_TASK),
            "pt_task/global_steps",
            self.report_wandb,
            self.hp_search_mode,
        )
        if self.hp_search_mode:
            self.best_metric = self.model.best_score  # type: ignore

    def train_doc_task(self, base_corpus: Section, extra_corpus: Corpus):
        self.__adapt_to_domain(extra_corpus, TrainTypes.DOC_TASK, lambda x: self.__create_and_save_model_from_scratch())

    def train_pt_doc_task(self, base_corpus: Section, extra_corpus: Corpus):
        self.__adapt_to_domain(extra_corpus, TrainTypes.PT_DOC_TASK, lambda x: self.pretrained_model)

    def train_bugs_task(self):
        self.__adapt_to_domain(
            self.tapt_data, TrainTypes.BUGS_TASK, lambda x: self.__create_and_save_model_from_scratch()
        )

    def train_pt_bugs_task(self):
        self.__adapt_to_domain(self.tapt_data, TrainTypes.PT_BUGS_TASK, lambda x: self.pretrained_model)

    def train_doc_bugs_task(self, extra_corpus: Corpus):
        pretraining_model_supplier = lambda x: self.__create_and_save_model_from_scratch()
        if self.start_train_from == "bugs":
            pretraining_model_supplier = lambda x: self.__get_doc_model_name(TrainTypes.DOC_BUGS_TASK, x)
            extra_corpus = self.tapt_data
        else:
            extra_corpus += self.tapt_data

        self.__adapt_to_domain(extra_corpus, TrainTypes.DOC_BUGS_TASK, pretraining_model_supplier)

    def train_pt_doc_bugs_task(self, extra_corpus: Corpus):
        pretraining_model_supplier = lambda x: self.pretrained_model
        if self.start_train_from == "bugs":
            pretraining_model_supplier = lambda x: self.__get_doc_model_name(TrainTypes.PT_DOC_BUGS_TASK, x)
            extra_corpus = self.tapt_data
        else:
            extra_corpus += self.tapt_data

        self.__adapt_to_domain(extra_corpus, TrainTypes.PT_DOC_BUGS_TASK, pretraining_model_supplier)

    def __get_doc_model_name(self, training_type: str, dapt_task_name: str):
        return os.path.join(self.save_to_path, self.name + "_" + dapt_task_name + "_" + training_type, "output_docs")

    def __adapt_to_domain(
        self, extra_corpus: Corpus, training_type: str, pretrained_model_supplier: Callable[[str], str]
    ):
        for dapt_task in self.domain_adaptation_tasks:
            self.logger.info(f"Start pre-training with {dapt_task.name} task")
            self.__do_adapt_to_domain(extra_corpus, dapt_task, pretrained_model_supplier(dapt_task.name), training_type)
            if self.hp_search_mode:
                self.best_metric += self.model.best_score / len(self.domain_adaptation_tasks)  # type: ignore
                wandb.log({"best_" + dapt_task.name + "_" + ValMetric.TASK: self.model.best_score})  # type: ignore
            self.logger.info(f"Train {training_type.replace('_', '+')} with {dapt_task.name} complete")

    def __do_adapt_to_domain(
        self,
        extra_corpus: Corpus,
        dapt_task: AbstractPreTrainingTask,
        pretrained_model: str,
        save_to_path_suffix: str,
    ):
        save_to_dir = os.path.join(self.save_to_path, self.name + "_" + dapt_task.name + "_" + save_to_path_suffix)
        word_embedding_model = (
            dapt_task.train_on_docs(
                pretrained_model,
                extra_corpus,
                self.task.evaluator,
                self.task.config.device,
                save_to_dir,
                self.report_wandb,
            )
            if self.start_train_from != "task"  # not self.start_train_from_task
            else dapt_task.load(save_to_dir)
        )

        self.model = self.task.train(
            word_embedding_model,
            save_to_dir,
            f"{dapt_task.name}_task/global_steps",
            self.report_wandb,
            self.hp_search_mode,
        )
        if not self.task.config.save_best_model:
            self.save(save_to_dir)

    def __create_and_save_model_from_scratch(self) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        bert_config = BertConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=self.task.config.max_len)
        dumb_model = BertModel(bert_config)

        dumb_model_name = tempfile.mkdtemp()
        tokenizer.save_pretrained(dumb_model_name)
        dumb_model.save_pretrained(dumb_model_name)

        return dumb_model_name

    def __init_wandb(self, name: str) -> Union[Run, RunDisabled, None]:
        return wandb.init(name=name, reinit=True, **self.wandb_config)  # type: ignore

    def train_and_save_all(self, base_corpus: Section, extra_corpus: Corpus, model_types_to_train: List[str]):
        run = None
        if TrainTypes.TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.TASK)
            self.train_task(base_corpus)
            self.logger.info(f"Train from scratch {self.name} SUCCESS")
            if not self.task.config.save_best_model:
                self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.TASK))
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.PT_TASK)
            self.train_pt_task(base_corpus)
            self.logger.info(f"Train pretrained {self.name} SUCCESS")
            if not self.task.config.save_best_model:
                self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.PT_TASK))
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.DOC_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.DOC_TASK)
            self.train_doc_task(base_corpus, extra_corpus)
            self.logger.info(f"Train DOC+TASK {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_DOC_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.PT_DOC_TASK)
            self.train_pt_doc_task(base_corpus, extra_corpus)
            self.logger.info(f"Train fine-tuned {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.BUGS_TASK)
            self.train_bugs_task()
            self.logger.info(f"Train {TrainTypes.BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.PT_BUGS_TASK)
            self.train_pt_bugs_task()
            self.logger.info(f"Train {TrainTypes.PT_BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.DOC_BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.DOC_BUGS_TASK)
            self.train_doc_bugs_task(extra_corpus)
            self.logger.info(f"Train {TrainTypes.DOC_BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_DOC_BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = self.__init_wandb(TrainTypes.PT_DOC_BUGS_TASK)
            self.train_pt_doc_bugs_task(extra_corpus)
            self.logger.info(f"Train {TrainTypes.PT_DOC_BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

    def get_embeddings(self, corpus: Section):
        return self.model.encode([" ".join(report) for report in corpus], show_progress_bar=True).astype(  # type: ignore
            np.float32
        )

    def get_doc_embedding(self, doc: List[str]):
        return self.get_embeddings([doc])[0]

    @classmethod
    def load(cls, path: str):
        sbert_model = SentenceTransformer(path)
        model = BertDomainModel()
        model.model = sbert_model
        return model
