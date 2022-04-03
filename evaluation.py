from abc import ABC

import numpy as np
import pandas as pd
import faiss

from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import FreqDist

from models import AbstractModel
from data_processing.util import get_corpus


class AbstractEvaluation:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test
        self.train_corpus = get_corpus(train)
        self.test_corpus = get_corpus(test)

        self.embeddings = None
        self.test_embs = None

    def evaluate(self, model: AbstractModel, topn=5):
        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)

        self.additional_preparation()

        test_size = 0
        TP = 0
        for ind, descr in enumerate(self.test_corpus):
            if self.test.iloc[ind]["id"] != self.test.iloc[ind]["disc_id"]:  # not in master_ids
                dupl_ids = self.get_dupl_ids(ind, topn)
                TP += np.any(self.train.iloc[dupl_ids]["disc_id"] == self.test.iloc[ind]["disc_id"])
                test_size += 1

            self.train = self.train.append(self.test.iloc[ind], ignore_index=True)
            self.update_history(ind)

        self.embeddings = None
        self.test_embs = None

        return TP / test_size

    def additional_preparation(self):
        raise NotImplementedError()

    def get_dupl_ids(self, query_num: int, topn: int):
        raise NotImplementedError()

    def update_history(self, query_num: int):
        raise NotImplementedError()


class SimpleEvaluation(AbstractEvaluation):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test)

    def additional_preparation(self):
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def get_dupl_ids(self, query_num, topn: int):
        return self.index.search(self.test_embs[query_num].reshape(1, -1), topn)[1][0]

    def update_history(self, query_num):
        tmp_emb = np.array([self.test_embs[query_num]])
        faiss.normalize_L2(tmp_emb)
        self.index.add(tmp_emb)


class TfIdfEvaluation(AbstractEvaluation):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, w: int):
        super(TfIdfEvaluation, self).__init__(train, test)
        self.w = w

        self.tf_idf = None
        self.train_tfidf_vectors = None
        self.test_tfidf_vectors = None

    def additional_preparation(self):
        self.tf_idf = TfidfVectorizer(use_idf=True)
        train_corpus_tf_idf = [" ".join(doc) for doc in self.train_corpus]
        self.tf_idf.fit(train_corpus_tf_idf)

        self.train_tfidf_vectors = self.tf_idf.transform(train_corpus_tf_idf)
        self.test_tfidf_vectors = self.tf_idf.transform([" ".join(doc) for doc in self.test_corpus])

    def get_dupl_ids(self, query_num, topn):
        sims_emb = cosine_similarity(self.embeddings, self.test_embs[query_num].reshape(1, -1)).squeeze()
        sims_tfidf = cosine_similarity(self.train_tfidf_vectors, self.test_tfidf_vectors[query_num]).squeeze()
        sims = self.w * sims_tfidf + (1 - self.w) * sims_emb
        return np.argsort(-sims)[:topn]

    def update_history(self, query_num):
        self.train_tfidf_vectors = vstack((self.train_tfidf_vectors, self.test_tfidf_vectors[query_num]))
        self.embeddings = np.append(self.embeddings, self.test_embs[query_num].reshape(1, -1), axis=0)


class IntersectionEvaluation(AbstractEvaluation):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, min_count: int):
        super(IntersectionEvaluation, self).__init__(train, test)
        self.min_count = min_count

    def additional_preparation(self):
        freq_dict = FreqDist()
        for report in self.train_corpus:
            freq_dict.update(report)
        for report in self.test_corpus:
            freq_dict.update(report)
        self.test_corpus = [
            list(filter(lambda x: freq_dict[x] >= self.min_count, report)) for report in self.test_corpus
        ]
        self.train_corpus = [
            list(filter(lambda x: freq_dict[x] >= self.min_count, report)) for report in self.train_corpus
        ]

    def get_dupl_ids(self, query_num, topn, **kwargs):
        counts = []
        for report in self.train_corpus:
            count = len(list(set(report) & set(self.test_corpus[query_num])))
            counts.append(count)
        return np.argsort(counts)[::-1][:topn]

    def update_history(self, query_num: int):
        self.train_corpus.append(self.test_corpus[query_num])

    class UtilModel(AbstractModel, ABC):
        def __init__(self):
            super(IntersectionEvaluation.UtilModel, self).__init__(0, 0)
            self.name = "Intersection Utility Model"

        def get_embeddings(self, corpus):
            return None
