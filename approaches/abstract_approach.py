from abc import ABC

import numpy as np
import pandas as pd

from text_models import AbstractModel
from data_processing.util import get_corpus


class AbstractApproach(ABC):
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
        for ind in range(len(self.test_corpus)):
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
