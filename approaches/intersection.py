import numpy as np
import pandas as pd
from nltk import FreqDist

from approaches import AbstractApproach
from text_models import AbstractModel


class IntersectionApproach(AbstractApproach):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, min_count: int):
        super(IntersectionApproach, self).__init__(train, test)
        self.min_count = min_count

    def setup_approach(self):
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

    def get_duplicated_ids(self, query_num, topn, **kwargs):
        counts = []
        for report in self.train_corpus:
            count = len(set(report) & set(self.test_corpus[query_num]))
            counts.append(count)
        return np.argsort(counts)[::-1][:topn]

    def update_history(self, query_num: int):
        self.train_corpus.append(self.test_corpus[query_num])

    class UtilModel(AbstractModel):
        def __init__(self):
            super(IntersectionApproach.UtilModel, self).__init__(0, 0)

        def get_embeddings(self, corpus):
            return None
