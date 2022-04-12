import json
import numpy as np

from nltk import FreqDist
from gensim.models.word2vec import Word2Vec

from data_processing.util import NumpyArrayEncoder
from text_models.abstract_model import AbstractModel


class RandomEmbeddingModel(AbstractModel):
    def __init__(self, train_corpus=None, vector_size=300, min_count=1, random_seed=42, w2v=False, save_to_path="./",
                 models_suffixes=None):
        super().__init__(vector_size=vector_size, seed=random_seed, save_to_path=save_to_path, models_suffixes=models_suffixes)
        self.min_count = min_count

        freq_dict = FreqDist()
        for docs in train_corpus:
            freq_dict.update(FreqDist(docs))

        dumb_w2v = None
        if w2v:
            dumb_w2v = Word2Vec(vector_size=self.vector_size, seed=random_seed, min_count=self.min_count)
            dumb_w2v.build_vocab(train_corpus)

        self.model = {}
        for word, freq in freq_dict.items():
            if freq >= self.min_count:
                self.model[word] = dumb_w2v.wv[word] if w2v else np.random.rand(self.vector_size)

    name = "random"

    def save(self, path):
        with open(path, "w+") as fp:
            json.dump(self.model, fp, cls=NumpyArrayEncoder)

    @classmethod
    def load(cls, path):
        with open(path) as json_file:
            model = json.load(json_file)
        for word, emb in model.items():
            model[word] = np.array(emb)

        random_model = RandomEmbeddingModel()
        random_model.model = model
        random_model.vector_size = len(list(model.values())[0])
        return random_model

    def train_from_scratch(self, corpus):
        pass

    def train_pretrained(self, corpus):
        pass

    def train_finetuned(self, base_corpus, extra_corpus):
        pass
