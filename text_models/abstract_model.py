import os
from abc import ABC

import numpy as np
from omegaconf import OmegaConf

import data_processing.util as util

config = OmegaConf.load(util.CONFIG_PATH)


class AbstractModel(ABC):
    name = "abstract"

    def __init__(self, vector_size=300, epochs=5, pretrained_model=None):
        self.vector_size = vector_size
        self.epochs = epochs
        self.model = None
        self.pretrained_model = pretrained_model

    def train_from_scratch(self, corpus):
        raise NotImplementedError()

    def train_pretrained(self, corpus):
        raise NotImplementedError()

    def train_finetuned(self, base_corpus, extra_corpus):
        self.train_pretrained(base_corpus + extra_corpus)

    def get_doc_embedding(self, doc):
        result = np.zeros(self.vector_size)
        size = 0
        for word in doc:
            if word in self.model:
                result += self.model[word]
                size += 1
        return result if size == 0 else result / size

    def get_embeddings(self, corpus):
        return np.array([self.get_doc_embedding(report) for report in corpus]).astype(np.float32)

    @classmethod
    def load(cls, path):
        raise NotImplementedError()

    def save(self, path):
        self.model.save(path)

    def train_and_save_all(self, base_corpus, extra_corpus):
        self.train_from_scratch(base_corpus)
        print(f"Train from scratch {self.name} SUCCESS")
        self.save(os.path.join(config.models_directory, self.name + config.models_suffixes.from_scratch))

        self.train_pretrained(base_corpus)
        print(f"Train pretrained {self.name} SUCCESS")
        self.save(os.path.join(config.models_directory, self.name + config.models_suffixes.pretrained))

        self.train_finetuned(base_corpus, extra_corpus)
        print(f"Train fine-tuned {self.name} SUCCESS")
        self.save(os.path.join(config.models_directory, self.name + config.models_suffixes.finetuned))
