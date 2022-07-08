import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Any

import numpy as np

from data_processing.util import Section, Corpus, fix_random_seed, flatten


class TrainTypes:
    TASK = "TASK"
    PT_TASK = "PT_TASK"
    DOC_TASK = "DOC_TASK"
    PT_DOC_TASK = "PT_DOC_TASK"


class AbstractModel(ABC):
    """
    Base class for all text models, that can be fine-tuned on docs and can also be used to map sentences / text to embeddings.

    :param vector_size: the size of embedding vector
    :param epochs: the number of train epochs
    :param pretrained_model: the name of pretrained text model
    :param seed: random seed
    :param save_to_path: where the trained model should be saved
    """

    name = "abstract"

    def __init__(
        self,
        vector_size: int = 300,
        epochs: int = 5,
        pretrained_model: str = "undefined",
        seed: int = 42,
        save_to_path: str = "./",
    ):
        self.logger = logging.getLogger(self.name)
        self.vector_size = vector_size
        self.epochs = epochs
        self.model: Optional[Any] = None
        self.pretrained_model = pretrained_model
        self.save_to_path = save_to_path

        fix_random_seed(seed)

    @abstractmethod
    def train_task(self, corpus: Section):
        """
        Train from scratch on the task of finding duplicate bug reports

        :param corpus: corpus of bug report descriptions
        """
        raise NotImplementedError()

    @abstractmethod
    def train_pt_task(self, corpus: Section):
        """
        Train on the task of finding duplicate bug reports using a pre-trained text model.

        :param corpus: corpus of bug report descriptions
        """
        raise NotImplementedError()

    def train_doc_task(self, base_corpus: Section, extra_corpus: Corpus):
        """
        Train from scratch on docs and then train on the task of finding duplicate bug reports.

        :param base_corpus: corpus of bug report descriptions
        :param extra_corpus: corpus of text artefacts
        """
        self.train_task(base_corpus + flatten(extra_corpus))  # type: ignore

    def train_pt_doc_task(self, base_corpus: Section, extra_corpus: Corpus):
        """
        Train on docs using a pre-trained model and then train on the task of finding duplicate bug reports.

        :param base_corpus: corpus of bug report descriptions
        :param extra_corpus: corpus of text artefacts
        """
        self.train_pt_task(base_corpus + flatten(extra_corpus))  # type: ignore

    def get_doc_embedding(self, doc: List[str]) -> np.ndarray:
        """
        Map one sentence/text to embedding

        :param doc: tokenized sentence/text
        :return: embedding
        """
        result = np.zeros(self.vector_size)
        size = 0
        for word in doc:
            if word in self.model:  # type: ignore
                result += self.model[word]  # type: ignore
                size += 1
        return result if size == 0 else result / size

    def get_embeddings(self, corpus: Section) -> np.ndarray:
        """
        Map corpus of sentences to embeddings

        :param corpus: tokenized sentences
        :return: embeddings
        """
        return np.array([self.get_doc_embedding(report) for report in corpus], dtype=np.float32)

    @classmethod
    def load(cls, path: str):
        """
        Loads model

        :param path: path on disk
        :return: AbstractModel that can be used to encode sentences / text
        """
        raise NotImplementedError()

    def save(self, path: str):
        """
        Saves model

        :param path: Path on disk
        """
        self.model.save(path)  # type: ignore

    def train_and_save_all(self, base_corpus: Section, extra_corpus: Corpus, model_types_to_train: List[str]):
        """
        Trains the model in various ways and save each resulting model.

        :param base_corpus: corpus of bug report descriptions
        :param extra_corpus: corpus of text artefacts
        :param model_types_to_train: list of training types
        """
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
