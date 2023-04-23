from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from approaches import AbstractApproach
from data_processing.util import get_corpus
from text_models import AbstractModel


class Metric:
    Acc = "Acc@k"
    MAP = "MAP@K"


class DuplicatesDetectionApproach(AbstractApproach, ABC):
    """
    Base class for all approaches to solving the problem of finding duplicate bug reports.
    This class also evaluates final metrics for all text models.

    :param train: Train dataset
    :param test: Test dataset
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test, [Metric.Acc, Metric.MAP])
        self.relevant_reports_num = self.__get_relevant_reports_num()

        self.train["id_num"] = np.arange(len(self.train.index))
        self.test["id_num"] = np.arange(len(self.test.index))

        self.train_corpus: List = get_corpus(train)
        self.test_corpus: List = get_corpus(test)

        self.test_size: Optional[int] = None
        self.true_positive_at_k: Optional[np.ndarray] = None

    def evaluate(self, model: AbstractModel, topks: List[int]) -> Dict[str, np.ndarray]:
        """
        Evaluate model.

        :param model: Evaluation model
        :param topks: What number of the most similar bug reports according to the model will be used in the evaluation
        :return: Acc@n for all topns from topns parameter.
        """
        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)
        self.train_copy = self.train.copy()

        self.setup_approach()

        self.test_size = 0
        self.true_positive_at_k = np.zeros(len(topks))
        self.avg_precision_at_k = np.zeros(len(topks))

        def eval_sample(query_report):
            if (
                query_report.id != query_report.disc_id and self.relevant_reports_num[query_report.id] > 0
            ):  # is duplicate
                dupl_ids = self.get_duplicated_ids(query_report.id_num, max(topks))
                for i, topk in enumerate(topks):

                    self.true_positive_at_k[i] += np.any(
                        self.train_copy.iloc[dupl_ids[:topk]]["disc_id"] == query_report.disc_id
                    )

                    sum_precisions = 0
                    num_correct = 0
                    for rank, dupl_id in enumerate(dupl_ids[:topk]):
                        if query_report.disc_id == self.train_copy.iloc[dupl_id]["disc_id"]:
                            num_correct += 1
                            sum_precisions += num_correct / (rank + 1)

                    avg_precision = sum_precisions / min(topk, self.relevant_reports_num[query_report.id])
                    self.avg_precision_at_k[i] += avg_precision

                self.test_size += 1
            self.train_copy = self.train_copy.append(query_report, ignore_index=True)
            self.update_history(query_report.id_num)

        self.test.apply(eval_sample, axis=1)

        metrics = {
            Metric.Acc: self.true_positive_at_k / self.test_size,
            Metric.MAP: self.avg_precision_at_k / self.test_size,
        }
        return metrics

    @abstractmethod
    def setup_approach(self):
        raise NotImplementedError()

    @abstractmethod
    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_history(self, query_num: int):
        raise NotImplementedError()

    def __get_relevant_reports_num(self) -> Counter:
        result: Counter = Counter()

        test_disc_ids = self.test.disc_id.tolist()
        count_train: Counter = Counter(self.train.disc_id)

        for i, query_id in enumerate(self.test.id):
            result[query_id] += count_train[test_disc_ids[i]]
            count_train[test_disc_ids[i]] += 1

        return result
