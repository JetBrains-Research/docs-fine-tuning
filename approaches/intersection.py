from abc import ABC

import numpy as np
import pandas as pd

from nltk import FreqDist

from text_models import AbstractModel
from approaches import AbstractApproach


class IntersectionApproach(AbstractApproach):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, min_count: int):
        super(IntersectionApproach, self).__init__(train, test)
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
            super(IntersectionApproach.UtilModel, self).__init__(0, 0)

        def get_embeddings(self, corpus):
            return None
