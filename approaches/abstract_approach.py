from abc import ABC

import numpy as np
import pandas as pd

from text_models import AbstractModel
from data_processing.util import get_corpus


class AbstractApproach(ABC):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test

        self.train["id_num"] = np.arange(len(self.train.index))
        self.test["id_num"] = np.arange(len(self.test.index))

        self.train_corpus = get_corpus(train)
        self.test_corpus = get_corpus(test)

        self.embeddings = None
        self.test_embs = None

    def evaluate(self, model: AbstractModel, topn=5):
        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)

        self.setup_approach()

        self.test_size = 0
        self.TP = 0

        def __eval_sample(query_report):
            if query_report.id != query_report.disc_id: # not in master ids
                dupl_ids = self.get_duplicated_ids(query_report.id_num, topn)
                self.TP += np.any(self.train.iloc[dupl_ids]["disc_id"] == query_report.disc_id)
                self.test_size += 1
            self.train = self.train.append(query_report, ignore_index=True)
            self.update_history(query_report.id_num)

        self.test.apply(__eval_sample, axis=1)

        self.embeddings = None
        self.test_embs = None

        return self.TP / self.test_size

    def setup_approach(self):
        raise NotImplementedError()

    def get_duplicated_ids(self, query_num: int, topn: int):
        raise NotImplementedError()

    def update_history(self, query_num: int):
        raise NotImplementedError()
