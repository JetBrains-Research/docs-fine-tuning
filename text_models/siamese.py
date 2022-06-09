import os.path
import tempfile
from typing import List, Union, Dict

import numpy as np
from gensim.test.utils import get_tmpfile
from omegaconf import DictConfig, ListConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, AutoTokenizer

from text_models import AbstractModel, TrainTypes
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks import tasks
from text_models.datasets import CosineSimilarityDataset, TripletDataset
from data_processing.util import Sentences, Sections


class BertSiameseModel(AbstractModel):
    def __init__(
        self,
        corpus: Sentences = None,
        disc_ids: List[str] = None,
        cnf_tasks: Union[DictConfig, ListConfig] = None,
        finetuning_strategies: List[str] = None,
        vector_size: int = 384,
        epochs: int = 5,
        batch_size: int = 16,
        n_examples: Union[str, int] = "all",
        max_len: int = 512,
        warmup_rate: float = 0.1,
        evaluation_steps: int = 500,
        evaluator_config: Union[DictConfig, ListConfig] = None,
        val_size: float = 0.1,
        task_loss: str = "cossim",  # or 'triplet'
        pretrained_model: str = "bert-base-uncased",
        tmp_file: str = get_tmpfile("pretrained_vectors.txt"),
        start_train_from_task: bool = False,
        device: str = "cpu",  # or 'cuda'
        save_best_model: bool = False,
        seed: int = 42,
        save_to_path: str = "./",
        models_suffixes: Union[Dict[str, str], DictConfig, ListConfig] = None,
    ):
        super().__init__(vector_size, epochs, pretrained_model, seed, save_to_path, models_suffixes)
        if finetuning_strategies is None:
            finetuning_strategies = ["mlm"]
        self.device = device
        self.tmp_file = tmp_file or get_tmpfile("pretrained_vectors.txt")
        self.batch_size = batch_size
        self.max_len = max_len
        self.loss = task_loss
        self.evaluation_steps = evaluation_steps
        self.save_best_model = save_best_model
        self.evaluator_config = evaluator_config
        self.start_train_from_task = start_train_from_task

        self.task_dataset = None
        self.evaluator = None
        self.vocab_size = None
        self.tokenizer = None
        self.warmup_steps = None

        if corpus is not None and disc_ids is not None:
            sentences = [" ".join(doc) for doc in corpus]
            train_size = int(len(corpus) * (1 - val_size))
            train_corpus = sentences[:train_size]
            train_disc_ids = disc_ids[:train_size]

            val_corpus = sentences[train_size:]
            val_disc_ids = disc_ids[train_size:]

            print(f"Train size = {train_size}, Validation size = {len(val_disc_ids)}")

            self.evaluator = self.__get_evaluator(train_corpus, train_disc_ids, val_corpus, val_disc_ids)
            self.task_dataset = self.__get_dataset(train_corpus, train_disc_ids, n_examples)

            self.n_examples = n_examples
            if n_examples == "all":
                self.n_examples = len(self.task_dataset)
            self.warmup_steps = np.ceil(self.n_examples * self.epochs * warmup_rate)

        if cnf_tasks is not None:
            self.finetuning_strategies = [tasks[name](**cnf_tasks[name]) for name in finetuning_strategies]

    name = "BERT_SIAMESE"

    def train_from_scratch(self, corpus: Sentences):
        dumb_model_name = self.__create_and_save_model_from_scratch()

        word_embedding_model = models.Transformer(dumb_model_name)
        self.__train_siamese(
            word_embedding_model, os.path.join(self.save_to_path, self.name + self.models_suffixes.from_scratch)
        )

    def train_pretrained(self, corpus: Sentences):
        word_embedding_model = models.Transformer(self.pretrained_model)
        self.__train_siamese(
            word_embedding_model, os.path.join(self.save_to_path, self.name + self.models_suffixes.pretrained)
        )

    def train_from_scratch_finetuned(self, base_corpus: Sentences, extra_corpus: Sections):
        for finetuning_task in self.finetuning_strategies:
            print(f"Start pretraining with {finetuning_task.name} task")
            dumb_model_name = self.__create_and_save_model_from_scratch()
            self.__train_finetuned_on_task(
                extra_corpus, finetuning_task, dumb_model_name, self.models_suffixes.doc_task
            )
            print(f"Train DOC+TASK with {finetuning_task.name} complete")

    def train_finetuned(self, base_corpus: Sentences, extra_corpus: Sections):
        for finetuning_task in self.finetuning_strategies:
            print(f"Start fine-tuning with {finetuning_task.name} task")
            self.__train_finetuned_on_task(
                extra_corpus, finetuning_task, self.pretrained_model, self.models_suffixes.finetuned
            )
            print(f"Train with {finetuning_task.name} complete")

    def __train_siamese(self, word_embedding_model: models.Transformer, save_to_dir: str):
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

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluator=self.evaluator,
            evaluation_steps=self.evaluation_steps,
            output_path=save_to_dir if self.save_best_model else os.path.join(save_to_dir, "output"),
            checkpoint_path=os.path.join(save_to_dir, "checkpoints"),
            show_progress_bar=True,
            checkpoint_save_total_limit=3,
            save_best_model=self.save_best_model,
        )

    def __train_finetuned_on_task(
        self,
        extra_corpus: List[List[List[str]]],
        finetuning_task: AbstractTask,
        pretrained_model: str,
        save_to_path_suffix: str,
    ):
        save_to_dir = os.path.join(self.save_to_path, self.name + "_" + finetuning_task.name + save_to_path_suffix)
        word_embedding_model = (
            finetuning_task.finetune_on_docs(
                pretrained_model,
                extra_corpus,
                self.evaluator,
                self.max_len,
                self.device,
                save_to_dir,
            )
            if not self.start_train_from_task
            else finetuning_task.load()
        )
        self.__train_siamese(word_embedding_model, save_to_dir)
        if not self.save_best_model:
            self.save(save_to_dir)

    def __get_dataset(self, corpus, disc_ids, n_examples):
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

        return InformationRetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            main_score_function="cos_sim",
            score_functions={"cos_sim": cos_sim},
            **self.evaluator_config,
        )

    def __create_and_save_model_from_scratch(self) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        bert_config = BertConfig(vocab_size=tokenizer.vocab_size, max_position_embeddings=self.max_len + 2)
        dumb_model = BertModel(bert_config)

        dumb_model_name = tempfile.mkdtemp()
        tokenizer.save_pretrained(dumb_model_name)
        dumb_model.save_pretrained(dumb_model_name)

        return dumb_model_name

    def train_and_save_all(self, base_corpus: Sentences, extra_corpus: Sections, model_types_to_train: List[str]):

        if TrainTypes.TASK in model_types_to_train:
            self.train_from_scratch(base_corpus)
            print(f"Train from scratch {self.name} SUCCESS")
            if not self.save_best_model:
                self.save(os.path.join(self.save_to_path, self.name + self.models_suffixes.from_scratch))

        if TrainTypes.PT_TASK in model_types_to_train:
            self.train_pretrained(base_corpus)
            print(f"Train pretrained {self.name} SUCCESS")
            if not self.save_best_model:
                self.save(os.path.join(self.save_to_path, self.name + self.models_suffixes.pretrained))

        if TrainTypes.DOC_TASK in model_types_to_train:
            self.train_from_scratch_finetuned(base_corpus, extra_corpus)
            print(f"Train DOC+TASK {self.name} SUCCESS")

        if TrainTypes.PT_DOC_TASK in model_types_to_train:
            self.train_finetuned(base_corpus, extra_corpus)
            print(f"Train fine-tuned {self.name} SUCCESS")

    def get_embeddings(self, corpus: Sentences):
        return self.model.encode([" ".join(report) for report in corpus], show_progress_bar=True).astype(np.float32)

    def get_doc_embedding(self, doc: List[str]):
        return self.get_embeddings([doc])[0]

    @classmethod
    def load(cls, path: str):
        sbert_model = SentenceTransformer(path)
        model = BertSiameseModel()
        model.model = sbert_model
        return model
