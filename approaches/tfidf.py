import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from approaches import AbstractApproach


class TfIdfApproach(AbstractApproach):
    """
    We build a term frequency-inverse document frequency(TF-IDF) matrix and take the sum of the metrics for text model
    embeddings and TF-IDF embeddings with coefficients w and (1 − w) as the final similarity metric.

    :param train: Train dataset
    :param test: Test dataset
    :param w: The weight of the text model embedding in similarity metric.
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, w: float = 0.3):
        super(TfIdfApproach, self).__init__(train, test)
        self.w = w

    def setup_approach(self):
        self.tf_idf = TfidfVectorizer(use_idf=True)
        train_corpus_tf_idf = [" ".join(doc) for doc in self.train_corpus]
        self.tf_idf.fit(train_corpus_tf_idf)

        self.train_tfidf_vectors = self.tf_idf.transform(train_corpus_tf_idf)
        self.test_tfidf_vectors = self.tf_idf.transform([" ".join(doc) for doc in self.test_corpus])

    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        sims_emb = cosine_similarity(self.embeddings, self.test_embs[query_num].reshape(1, -1)).squeeze()
        sims_tfidf = cosine_similarity(self.train_tfidf_vectors, self.test_tfidf_vectors[query_num]).squeeze()
        sims = self.w * sims_tfidf + (1 - self.w) * sims_emb
        return np.argsort(-sims)[:topn]

    def update_history(self, query_num: int):
        self.train_tfidf_vectors = vstack((self.train_tfidf_vectors, self.test_tfidf_vectors[query_num]))
        self.embeddings = np.append(self.embeddings, self.test_embs[query_num].reshape(1, -1), axis=0)
