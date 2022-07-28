import json
from typing import Optional

import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk import FreqDist

from data_processing.util import NumpyArrayEncoder, Section
from text_models.abstract_model import AbstractModel


class RandomEmbeddingModel(AbstractModel):
    """
    Baseline text model that map every word to random embedding

    :param train_corpus: Sentences from which random embeddings will be generated
    :param vector_size: The size of embedding vector
    :param min_count: Ignores all words with total frequency lower than this
    :param seed: Random seed
    :param rand_by_w2v: Use dumb Word2Vec model to generate random embeddings
    :param save_to_path: Where the trained model should be saved
    """

    def __init__(
        self,
        train_corpus: Optional[Section] = None,
        vector_size: int = 300,
        min_count: int = 1,
        seed: int = 42,
        rand_by_w2v: bool = False,
        save_to_path: str = "./",
    ):
        super().__init__(vector_size=vector_size, seed=seed, save_to_path=save_to_path)
        self.min_count = min_count

        if train_corpus is None:
            return

        freq_dict = FreqDist()
        for docs in train_corpus:
            freq_dict.update(FreqDist(docs))

        dumb_w2v = None
        if rand_by_w2v:
            dumb_w2v = Word2Vec(vector_size=self.vector_size, seed=seed, min_count=self.min_count)
            dumb_w2v.build_vocab(train_corpus)

        self.model = {}
        for word, freq in freq_dict.items():
            if freq >= self.min_count:
                self.model[word] = dumb_w2v.wv[word] if rand_by_w2v else np.random.rand(self.vector_size)  # type: ignore

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
