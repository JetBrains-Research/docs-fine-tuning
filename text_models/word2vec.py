import gensim.downloader as api
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import get_tmpfile

from data_processing.util import Section
from text_models.abstract_model import AbstractModel


class W2VModel(AbstractModel):
    """
    Word2Vec model, that can be fine-tuned on docs and can also be used to map sentences / text to embeddings.

    :param vector_size: The size of embedding vector
    :param epochs: The number of train epochs
    :param min_count: Ignores all words with total frequency lower than this
    :param pretrained_model: The name of pretrained text model
    :param tmp_file: Temporary file that will be used to store the model for fine-tuning
    :param seed: Random seed
    :param save_to_path: Where the trained model should be saved
    """

    name = "Word2Vec"

    def __init__(
        self,
        vector_size: int = 300,
        epochs: int = 5,
        min_count: int = 1,
        pretrained_model: str = "word2vec-google-news-300",
        tmp_file: str = get_tmpfile("pretrained_vectors.txt"),
        save_to_path: str = "./",
    ):
        super().__init__(vector_size, epochs, pretrained_model, save_to_path)
        self.tmp_file = tmp_file or get_tmpfile("pretrained_vectors.txt")
        self.init_vocab = self.__get_init_vocab()
        self.min_count = min_count

    def train_task(self, corpus: Section):
        self.model = Word2Vec(corpus, vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)

    def train_pt_task(self, corpus: Section):
        if self.init_vocab is None:
            raise RuntimeError("Init vocab is None")

        self.model = Word2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)
        self.model.build_vocab(corpus)
        self.model.min_count = 1
        self.model.build_vocab(self.init_vocab, update=True)
        self.model.wv.vectors_lockf = np.ones(len(self.model.wv))
        self.model.wv.intersect_word2vec_format(self.tmp_file, binary=False, lockf=1.0)
        self.model.min_count = self.min_count
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    def __get_init_vocab(self):
        pretrained = api.load(self.pretrained_model)
        pretrained.save_word2vec_format(self.tmp_file)
        return [list(pretrained.key_to_index.keys())]

    @classmethod
    def load(cls, path):
        loaded_model = Word2Vec.load(path)
        created_model = W2VModel(loaded_model.vector_size, loaded_model.epochs, loaded_model.min_count)
        created_model.model = loaded_model.wv
        return created_model
