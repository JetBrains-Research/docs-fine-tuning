import json

import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk import FreqDist

from data_processing.util import NumpyArrayEncoder, Section
from text_models.abstract_model import AbstractModel


class RandomEmbeddingModel(AbstractModel):
    def __init__(
        self,
        train_corpus: Section = None,
        vector_size: int = 300,
        min_count: int = 1,
        random_seed: int = 42,
        rand_by_w2v: bool = False,
        save_to_path: str = "./",
    ):
        super().__init__(vector_size=vector_size, seed=random_seed, save_to_path=save_to_path)
        self.min_count = min_count

        freq_dict = FreqDist()
        for docs in train_corpus:
            freq_dict.update(FreqDist(docs))

        dumb_w2v = None
        if rand_by_w2v:
            dumb_w2v = Word2Vec(vector_size=self.vector_size, seed=random_seed, min_count=self.min_count)
            dumb_w2v.build_vocab(train_corpus)

        self.model = {}
        for word, freq in freq_dict.items():
            if freq >= self.min_count:
                self.model[word] = dumb_w2v.wv[word] if rand_by_w2v else np.random.rand(self.vector_size)

    name = "random"

    def save(self, path: str):
        with open(path, "w+") as fp:
            json.dump(self.model, fp, cls=NumpyArrayEncoder)

    @classmethod
    def load(cls, path: str):
        with open(path) as json_file:
            model = json.load(json_file)
        for word, emb in model.items():
            model[word] = np.array(emb)

        random_model = RandomEmbeddingModel()
        random_model.model = model
        random_model.vector_size = len(list(model.values())[0])
        return random_model

    def train_task(self, corpus: Section):
        pass

    def train_pt_task(self, corpus: Section):
        pass
