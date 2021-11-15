import os.path

import numpy as np
import csv

import gensim.downloader as api

from gensim.test.utils import get_tmpfile
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer


class AbstractModel:
    def __init__(self, model=None):
        self.model = model
        self.name = "abstract"

    def train_random(self, corpus):
        raise NotImplementedError()

    def train_pretrained(self, corpus):
        raise NotImplementedError()

    def train_finetuned(self, base_corpus, extra_corpus):
        self.train_pretrained(base_corpus + extra_corpus)

    def save(self, path):
        self.model.save(path)

    def train_and_save_all(self, base_corpus, extra_corpus):
        self.train_random(base_corpus)
        print(f"Train random {self.name} SUCCESS")
        self.save(os.path.join("saved_models", f"{self.name}_random.model"))

        self.train_pretrained(base_corpus)
        print(f"Train pretrained {self.name} SUCCESS")
        self.save(os.path.join("saved_models", f"{self.name}_pretrained.model"))

        self.train_finetuned(base_corpus, extra_corpus)
        print(f"Train fine-tuned {self.name} SUCCESS")
        self.save(os.path.join("saved_models", f"{self.name}_finetuned.model"))


class W2VModel(AbstractModel):
    def __init__(self, vector_size=300, epochs=5, tmp_file=get_tmpfile("pretrained_vectors.txt")):
        super().__init__()
        self.vector_size = vector_size
        self.name = "w2v"
        self.epochs = epochs
        self.tmp_file = tmp_file
        self.init_vocab = self.__get_init_vocab()

    def train_random(self, corpus):
        self.model = Word2Vec(corpus, vector_size=self.vector_size, min_count=1, epochs=self.epochs)

    def train_pretrained(self, corpus):
        if self.init_vocab is None:
            raise RuntimeError("Init vocab is None")

        self.model = Word2Vec(vector_size=self.vector_size, min_count=1, epochs=self.epochs)
        self.model.build_vocab(corpus)
        self.model.build_vocab(self.init_vocab, update=True)
        self.model.wv.vectors_lockf = np.ones(len(self.model.wv))
        self.model.wv.intersect_word2vec_format(self.tmp_file, binary=False, lockf=1.0)

        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    def __get_init_vocab(self):
        pretrained = api.load(f"glove-wiki-gigaword-{self.vector_size}")  # TODO: change to 'word2vec-google-news-300'
        pretrained.save_word2vec_format(self.tmp_file)
        return [list(pretrained.key_to_index.keys())]


class FastTextModel(AbstractModel):
    def __init__(self, vector_size=300, epochs=5):
        super().__init__()
        self.name = "ft"
        self.vector_size = vector_size
        self.epochs = epochs

    def train_random(self, corpus):
        self.model = FastText(corpus, vector_size=self.vector_size, min_count=1, epochs=self.epochs)

    def train_pretrained(self, corpus):
        self.model = load_facebook_model(os.path.join("pretrained_models", "cc.en.300.bin"))
        self.model.build_vocab(corpus, update=True)
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)


# TODO: save embeddings format and build vocabulary
class GloveModel(AbstractModel):
    def __init__(self, corpus=None, vector_size=300, max_iter=100):
        super().__init__()
        self.name = "glove"
        self.vector_size = vector_size
        self.max_iter = max_iter
        self.vocab = self.__get_vocab(corpus)
        self.cooccurrence = self.__get_cooccurrance_matrix(corpus)
        self.pretrained = self.__glove2dict(os.path.join("pretrained_models", f"glove.6B.{self.vector_size}d.txt"))

        self.keyed_vectors = None

    def train_random(self, corpus):
        self.model = GloVe(n=self.vector_size, max_iter=self.max_iter)
        embeddings = self.model.fit(self.cooccurrence)
        self.keyed_vectors = self.__embeddings2model(embeddings)

    def train_pretrained(self, corpus):
        self.model = Mittens(n=self.vector_size, max_iter=self.max_iter)
        embeddings = self.model.fit(self.cooccurrence, vocab=self.vocab, initial_embedding_dict=self.pretrained)
        self.keyed_vectors = self.__embeddings2model(embeddings)

    def train_finetuned(self, base_corpus, extra_corpus):
        full_corpus = base_corpus + extra_corpus
        self.cooccurrence = self.__get_cooccurrance_matrix(full_corpus)
        self.vocab = self.__get_vocab(full_corpus)
        self.train_pretrained(full_corpus)

    def save(self, path):
        self.keyed_vectors.save(path)

    def __get_cooccurrance_matrix(self, corpus):
        # TODO: get matrix with sliding window
        docs = [' '.join(doc) for doc in corpus]
        cv = CountVectorizer(ngram_range=(1, 1), vocabulary=self.vocab)
        X = cv.fit_transform(docs)
        Xc = (X.T * X)
        Xc.setdiag(0)
        return Xc.toarray()

    def __embeddings2model(self, embeddings):
        word_dict = dict(zip(self.vocab, embeddings))
        vocab_size = len(word_dict)
        result = KeyedVectors(self.vector_size)
        result.vectors = np.array([np.array(v) for v in word_dict.values()])

        for i, word in enumerate(word_dict.keys()):
            result.vocab[word] = Vocab(index=i, count=vocab_size - i)
            result.index2word.append(word)
        return result

    @staticmethod
    def __glove2dict(glove_filename):
        with open(glove_filename, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
            embed = {line[0]: np.array(list(map(float, line[1:])))
                     for line in reader}
        return embed

    # TODO: OOV
    @staticmethod
    def __get_vocab(corpus):
        flatten_corpus = [token for doc in corpus for token in doc]
        return list(set(flatten_corpus))

