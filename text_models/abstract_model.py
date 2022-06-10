import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Union
from omegaconf import DictConfig, ListConfig

from data_processing.util import Sentences, Sections

import numpy as np


class TrainTypes:
    TASK = "TASK"
    PT_TASK = "PT+TASK"
    DOC_TASK = "DOC+TASK"
    PT_DOC_TASK = "PT+DOC+TASK"


class AbstractModel(ABC):
    name = "abstract"

    def __init__(
        self,
        vector_size: int = 300,
        epochs: int = 5,
        pretrained_model: str = None,
        seed: int = 42,
        save_to_path: str = "./",
        models_suffixes: Union[Dict[str, str], DictConfig, ListConfig] = None,
    ):
        if models_suffixes is None:
            models_suffixes = {
                "from_scratch": "_task.model",
                "pretrained": "_pt_task.model",
                "finetuned": "_pt_doc_task.model",
                "doc_task": "_doc_task.model",
            }

        self.logger = logging.getLogger(self.name)
        self.vector_size = vector_size
        self.epochs = epochs
        self.model = None
        self.pretrained_model = pretrained_model
        self.save_to_path = save_to_path
        self.models_suffixes = models_suffixes
        np.random.seed(seed)

    @abstractmethod
    def train_from_scratch(self, corpus: Sentences):
        raise NotImplementedError()

    @abstractmethod
    def train_pretrained(self, corpus: Sentences):
        raise NotImplementedError()

    def train_from_scratch_finetuned(self, base_corpus: Sentences, extra_corpus: Union[Sentences, Sections]):
        self.train_from_scratch(base_corpus + extra_corpus)

    def train_finetuned(self, base_corpus: Sentences, extra_corpus: Union[Sentences, Sections]):
        self.train_pretrained(base_corpus + extra_corpus)

    def get_doc_embedding(self, doc: List[str]):
        result = np.zeros(self.vector_size)
        size = 0
        for word in doc:
            if word in self.model:
                result += self.model[word]
                size += 1
        return result if size == 0 else result / size

    def get_embeddings(self, corpus: Sentences):
        return np.array([self.get_doc_embedding(report) for report in corpus]).astype(np.float32)

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        self.model.save(path)

    def train_and_save_all(
        self, base_corpus: Sentences, extra_corpus: Union[Sentences, Sections], model_types_to_train: List[str]
    ):

        if TrainTypes.TASK in model_types_to_train:
            self.train_from_scratch(base_corpus)
            self.logger.info(f"Train from scratch(TASK) {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + self.models_suffixes["from_scratch"]))

        if TrainTypes.PT_TASK in model_types_to_train:
            self.train_pretrained(base_corpus)
            self.logger.info(f"Train pretrained(PT+TASK) {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + self.models_suffixes["pretrained"]))

        if TrainTypes.DOC_TASK in model_types_to_train:
            self.train_from_scratch_finetuned(base_corpus, extra_corpus)
            self.logger.info(f"Train DOC+TASK {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + self.models_suffixes["doc_task"]))

        if TrainTypes.PT_DOC_TASK in model_types_to_train:
            self.train_finetuned(base_corpus, extra_corpus)
            self.logger.info(f"Train fine-tuned(PT+DOC+TASK) {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + self.models_suffixes["finetuned"]))
