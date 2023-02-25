import os.path
import tempfile
from typing import List, Union, Callable, Optional

import numpy as np
import torch.utils.data
import wandb
from gensim.test.utils import get_tmpfile
from omegaconf import DictConfig, ListConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, AutoTokenizer
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from data_processing.util import Section, Corpus, flatten
from text_models import AbstractModel, TrainTypes
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks import tasks
from text_models.bert_tasks.evaluation import WandbLoggingEvaluator
from text_models.datasets import CosineSimilarityDataset, TripletDataset


class BertSiameseModel(AbstractModel):
    """
    Text Transformer-Based model, that can be fine-tuned on docs through various tasks
    and can also be used to map sentences / text to embeddings.
    Model uses Siamese Neural Network (SNN) architecture for the task of finding duplicate bug reports.

    :param corpus: Corpus of bug descriptions for dataset building
    :param disc_ids: List where position i is the id of the oldest bug report that is a duplicate of i-th bug report
    :param finetuning_strategies: List of fine-tuning tasks that will be used for fine-tuning on docs
    :param cnf_tasks: Configuration for fine-tuning tasks
    :param vector_size: The size of embedding vector
    :param epochs: Number of train epochs
    :param batch_size: Batch size used for SNN training
    :param n_examples: Number of bug report pairs that will be used for SNN training
    :param max_len: max_length parameter of tokenizer
    :param warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
    :param evaluation_steps: Number of update steps between two evaluations
    :param evaluator_config: Configuration for the evaluator
    :param val_size: Ratio of total training examples used for validation dataset
    :param task_loss: The loss function for SNN. Possible values: "cossim", "triplet"
    :param pretrained_model: The name of pretrained text model
    :param start_train_from_task: If True then fine-tuning on docs step will be skipped and fine-tuned model
                                  from save_to_path will be used to train SNN as pre-trained model.
    :param device: What device will be used for training. Possible values: "cpu", "cuda".
    :param save_best_model: Whether or not to save the best model found during training at the end of training.
    :param seed: Random seed
    :param save_to_path: Where the trained model should be saved
    """

    def __init__(
        self,
        corpus: Corpus = None,
        disc_ids: List[str] = None,
        cnf_tasks: Union[DictConfig, ListConfig] = None,
        finetuning_strategies: List[str] = None,
        vector_size: int = 384,
        epochs: int = 5,
        batch_size: int = 16,
        n_examples: Union[str, int] = "all",
        max_len: int = None,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        evaluation_steps: Optional[int] = None,  # if None then epoch mode will be used
        save_steps: Optional[int] = None,  # if None then epoch mode will be used
        evaluator_config: Union[DictConfig, ListConfig] = None,
        val_size: float = 0.1,
        task_loss: str = "cossim",  # or 'triкplet'
        pretrained_model: str = "bert-base-uncased",
        tmp_file: str = get_tmpfile("pretrained_vectors.txt"),
        start_train_from_task: bool = False,
        start_train_from_bugs: bool = False,
        device: str = "cpu",  # or 'cuda'
        save_best_model: bool = False,
        seed: int = 42,
        save_to_path: str = "./",
        report_wandb: bool = False,
    ):
        super().__init__(vector_size, epochs, pretrained_model, seed, save_to_path)
        if finetuning_strategies is None:
            finetuning_strategies = ["mlm"]
        self.device = device
        self.tmp_file = tmp_file or get_tmpfile("pretrained_vectors.txt")
        self.batch_size = batch_size
        self.max_len = max_len
        self.loss = task_loss
        self.evaluation_steps = evaluation_steps
        self.save_steps = save_steps
        self.save_best_model = save_best_model
        self.evaluator_config = evaluator_config
        self.start_train_from_task = start_train_from_task
        self.start_train_from_bugs = start_train_from_bugs
        self.report_wandb = report_wandb
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        self.vocab_size = None
        self.tokenizer = None
        self.tapt_data: Corpus = [[[]]]

        if corpus is not None and disc_ids is not None:
            sentences = [" ".join(doc) for doc in list(map(flatten, corpus))]
            train_size = int(len(sentences) * (1 - val_size))
            train_corpus = sentences[:train_size]
            train_disc_ids = disc_ids[:train_size]

            val_corpus = sentences[train_size:]
            val_disc_ids = disc_ids[train_size:]

            self.logger.info(f"Train size = {train_size}, Validation size = {len(val_disc_ids)}")

            self.evaluator = self.__get_evaluator(train_corpus, train_disc_ids, val_corpus, val_disc_ids)
            self.task_dataset = self.__get_dataset(train_corpus, train_disc_ids, n_examples)

            self.tapt_data = corpus[:train_size]

            self.n_examples = len(self.task_dataset) if n_examples == "all" else int(n_examples)  # type: ignore

        if cnf_tasks is not None:
            self.finetuning_strategies = [
                tasks[name](**cnf_tasks[name]) for name in finetuning_strategies  # type: ignore
            ]

    name = "BERT_SIAMESE"

    def train_task(self, corpus: Section):
        dumb_model_name = self.__create_and_save_model_from_scratch()

        word_embedding_model = models.Transformer(dumb_model_name, max_seq_length=self.max_len)
        self.__train_siamese(
            word_embedding_model,
            os.path.join(self.save_to_path, self.name + "_" + TrainTypes.TASK),
            "task/global_steps",
        )

    def train_pt_task(self, corpus: Section):
        word_embedding_model = models.Transformer(self.pretrained_model, max_seq_length=self.max_len)
        self.__train_siamese(
            word_embedding_model,
            os.path.join(self.save_to_path, self.name + "_" + TrainTypes.PT_TASK),
            "pt_task/global_steps",
        )

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
        if not self.start_train_from_task and self.start_train_from_bugs:
            pretraining_model_supplier = lambda x: self.__get_doc_model_name(TrainTypes.DOC_BUGS_TASK, x)
            extra_corpus = self.tapt_data
        else:
            extra_corpus += self.tapt_data

        self.__adapt_to_domain(extra_corpus, TrainTypes.DOC_BUGS_TASK, pretraining_model_supplier)

    def train_pt_doc_bugs_task(self, extra_corpus: Corpus):
        pretraining_model_supplier = lambda x: self.pretrained_model
        if not self.start_train_from_task and self.start_train_from_bugs:
            pretraining_model_supplier = lambda x: self.__get_doc_model_name(TrainTypes.PT_DOC_BUGS_TASK, x)
            extra_corpus = self.tapt_data
        else:
            extra_corpus += self.tapt_data

        self.__adapt_to_domain(extra_corpus, TrainTypes.PT_DOC_BUGS_TASK, pretraining_model_supplier)

    def __get_doc_model_name(self, training_type: str, finetuning_task_name: str):
        return os.path.join(
            self.save_to_path, self.name + "_" + finetuning_task_name + "_" + training_type, "output_docs"
        )

    def __adapt_to_domain(
        self, extra_corpus: Corpus, training_type: str, pretrained_model_supplier: Callable[[str], str]
    ):
        for finetuning_task in self.finetuning_strategies:
            self.logger.info(f"Start pre-training with {finetuning_task.name} task")
            self.__train_finetuned_on_task(
                extra_corpus, finetuning_task, pretrained_model_supplier(finetuning_task.name), training_type
            )
            self.logger.info(f"Train {training_type.replace('_', '+')} with {finetuning_task.name} complete")

    def __train_siamese(
        self, word_embedding_model: models.Transformer, save_to_dir: str, step_metric: Optional[str] = None
    ):
        word_embedding_model.max_seq_length = self.max_len
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.vector_size,
            activation_function=nn.Tanh(),
        )
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)

        train_dataloader = DataLoader(self.task_dataset, shuffle=True, batch_size=self.batch_size)
        train_loss = (
            losses.CosineSimilarityLoss(self.model)
            if self.loss == "cossim"
            else losses.TripletLoss(
                model=self.model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=1
            )
        )

        evaluator = (
            WandbLoggingEvaluator(self.evaluator, step_metric, len(train_dataloader))
            if self.report_wandb and step_metric is not None
            else self.evaluator
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=np.ceil(len(train_dataloader) * self.epochs * self.warmup_ratio),
            weight_decay=self.weight_decay,
            evaluator=evaluator,
            evaluation_steps=0 if self.evaluation_steps is None else self.evaluation_steps,
            output_path=save_to_dir if self.save_best_model else os.path.join(save_to_dir, "output"),
            checkpoint_path=os.path.join(save_to_dir, "checkpoints"),
            show_progress_bar=True,
            checkpoint_save_steps=self.save_steps,
            save_best_model=self.save_best_model,
        )

    def __train_finetuned_on_task(
        self,
        extra_corpus: Corpus,
        finetuning_task: AbstractTask,
        pretrained_model: str,
        save_to_path_suffix: str,
    ):
        save_to_dir = os.path.join(
            self.save_to_path, self.name + "_" + finetuning_task.name + "_" + save_to_path_suffix
        )
        word_embedding_model = (
            finetuning_task.finetune_on_docs(
                pretrained_model,
                extra_corpus,
                self.evaluator,
                self.device,
                save_to_dir,
                self.report_wandb,
            )
            if not self.start_train_from_task
            else finetuning_task.load(save_to_dir)
        )

        self.__train_siamese(word_embedding_model, save_to_dir, f"{finetuning_task.name}_task/global_steps")
        if not self.save_best_model:
            self.save(save_to_dir)

    def __get_dataset(self, corpus, disc_ids, n_examples) -> torch.utils.data.Dataset:
        if self.loss == "cossim":
            return CosineSimilarityDataset(corpus, disc_ids, n_examples, shuffle=True)
        if self.loss == "triplet":
            return TripletDataset(corpus, disc_ids, n_examples, shuffle=True)
        raise ValueError("Unsupported loss")

    def __get_evaluator(
        self, train_corpus: List[str], train_disc_ids: List[str], val_corpus: List[str], val_disc_ids: List[str]
    ) -> InformationRetrievalEvaluator:
        queries = {qid: query for qid, query in enumerate(val_corpus)}
        corpus = {cid: doc for cid, doc in enumerate(train_corpus)}
        relevant_docs = {
            qid: {cid for cid in corpus.keys() if train_disc_ids[cid] == val_disc_ids[qid]} for qid in queries.keys()
        }

        evaluator = InformationRetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            main_score_function="cos_sim",
            score_functions={"cos_sim": cos_sim},  # type: ignore
            show_progress_bar=True,
            **self.evaluator_config,
        )
        evaluator.metrics = None
        evaluator.val_dataset = val_corpus
        return evaluator

    def __create_and_save_model_from_scratch(self) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        bert_config = BertConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=self.max_len)
        dumb_model = BertModel(bert_config)

        dumb_model_name = tempfile.mkdtemp()
        tokenizer.save_pretrained(dumb_model_name)
        dumb_model.save_pretrained(dumb_model_name)

        return dumb_model_name

    @staticmethod
    def __init_wandb(name: str) -> Union[Run, RunDisabled, None]:
        return wandb.init(project="docs-fine-tuning", entity="jbr-docs-fine-tuning", name=name, reinit=True)

    def train_and_save_all(self, base_corpus: Section, extra_corpus: Corpus, model_types_to_train: List[str]):
        run = None
        if TrainTypes.TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.TASK)
            self.train_task(base_corpus)
            self.logger.info(f"Train from scratch {self.name} SUCCESS")
            if not self.save_best_model:
                self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.TASK))
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.PT_TASK)
            self.train_pt_task(base_corpus)
            self.logger.info(f"Train pretrained {self.name} SUCCESS")
            if not self.save_best_model:
                self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.PT_TASK))
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.DOC_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.DOC_TASK)
            self.train_doc_task(base_corpus, extra_corpus)
            self.logger.info(f"Train DOC+TASK {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_DOC_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.PT_DOC_TASK)
            self.train_pt_doc_task(base_corpus, extra_corpus)
            self.logger.info(f"Train fine-tuned {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.BUGS_TASK)
            self.train_bugs_task()
            self.logger.info(f"Train {TrainTypes.BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.PT_BUGS_TASK)
            self.train_pt_bugs_task()
            self.logger.info(f"Train {TrainTypes.PT_BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.DOC_BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.DOC_BUGS_TASK)
            self.train_doc_bugs_task(extra_corpus)
            self.logger.info(f"Train {TrainTypes.DOC_BUGS_TASK.replace('_', '+')} {self.name} SUCCESS")
            if self.report_wandb and run is not None:
                run.finish()

        if TrainTypes.PT_DOC_BUGS_TASK in model_types_to_train:
            if self.report_wandb:
                run = BertSiameseModel.__init_wandb(TrainTypes.PT_DOC_BUGS_TASK)
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
        model = BertSiameseModel()
        model.model = sbert_model
        return model
