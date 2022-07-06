import logging
import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional

import numpy as np

from data_processing.util import Section, Corpus, fix_random_seed


class TrainTypes:
    TASK = "TASK"
    PT_TASK = "PT_TASK"
    DOC_TASK = "DOC_TASK"
    PT_DOC_TASK = "PT_DOC_TASK"


class AbstractModel(ABC):
    name = "abstract"

    def __init__(
        self,
        vector_size: int = 300,
        epochs: int = 5,
        pretrained_model: Optional[str] = None,
        seed: int = 42,
        save_to_path: str = "./",
    ):
        self.logger = logging.getLogger(self.name)
        self.vector_size = vector_size
        self.epochs = epochs
        self.model = None
        self.pretrained_model = pretrained_model
        self.save_to_path = save_to_path

        fix_random_seed(seed)

    @abstractmethod
    def train_task(self, corpus: Section):
        raise NotImplementedError()

    @abstractmethod
    def train_pt_task(self, corpus: Section):
        raise NotImplementedError()

    def train_doc_task(self, base_corpus: Section, extra_corpus: Union[Section, Corpus]):
        self.train_task(base_corpus + extra_corpus)

    def train_pt_doc_task(self, base_corpus: Section, extra_corpus: Union[Section, Corpus]):
        self.train_pt_task(base_corpus + extra_corpus)

    def get_doc_embedding(self, doc: List[str]):
        result = np.zeros(self.vector_size)
        size = 0
        for word in doc:
            if word in self.model:
                result += self.model[word]
                size += 1
        return result if size == 0 else result / size

    def get_embeddings(self, corpus: Section):
        return np.array([self.get_doc_embedding(report) for report in corpus], dtype=np.float32)

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError()

    def save(self, path: str):
        self.model.save(path)

    def train_and_save_all(
        self, base_corpus: Section, extra_corpus: Union[Section, Corpus], model_types_to_train: List[str]
    ):

        if TrainTypes.TASK in model_types_to_train:
            self.train_task(base_corpus)
            self.logger.info(f"Train TASK {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.TASK))

        if TrainTypes.PT_TASK in model_types_to_train:
            self.train_pt_task(base_corpus)
            self.logger.info(f"Train PT+TASK {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.PT_TASK))

        if TrainTypes.DOC_TASK in model_types_to_train:
            self.train_doc_task(base_corpus, extra_corpus)
            self.logger.info(f"Train DOC+TASK {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.DOC_TASK))

        if TrainTypes.PT_DOC_TASK in model_types_to_train:
            self.train_pt_doc_task(base_corpus, extra_corpus)
            self.logger.info(f"Train PT+DOC+TASK {self.name} SUCCESS")
            self.save(os.path.join(self.save_to_path, self.name + "_" + TrainTypes.PT_DOC_TASK))
