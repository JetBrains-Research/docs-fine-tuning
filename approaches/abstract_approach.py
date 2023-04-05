import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Type
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.util import get_corpus
from text_models import AbstractModel


class Metric:
    Acc = "Acc@k"
    MAP = "MAP@K"


class AbstractApproach(ABC):
    """
    Base class for all approaches to solving the problem of finding duplicate bug reports.
    This class also evaluates final metrics for all text models.

    :param train: Train dataset
    :param test: Test dataset
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test
        self.relevant_reports_num = self.__get_relevant_reports_num()

        self.train["id_num"] = np.arange(len(self.train.index))
        self.test["id_num"] = np.arange(len(self.test.index))

        self.train_corpus: List = get_corpus(train)
        self.test_corpus: List = get_corpus(test)

        self.results: Optional[pd.DataFrame] = None
        self.test_size: Optional[int] = None
        self.true_positive_at_k: Optional[np.ndarray] = None

    def evaluate_all(
        self,
        model_types: List[str],
        model_class: Type[AbstractModel],
        models_directory: str,
        topns: List[int],
        verbose: bool = True,
    ):
        """
        Evaluate all models trained in different ways.

        :param model_types: model types to evaluate
        :param model_class: model class to load
        :param models_directory: Which directory use to load models
        :param topns: What number of the most similar bug reports according to the model will be used in the evaluation
        :param verbose: Should evaluation logs be verbose or not
        """

        res_dict: Dict[str, Union[np.ndarray, List[int]]] = {"k": topns}
        models_dict = self._load_models(model_types, model_class, models_directory)
        for name, model in models_dict.items():
            if model is not None:
                metrics = self.evaluate(model, topns)
                for metric_name, values in metrics.items():
                    res_dict[name + "_" + metric_name] = values

        self.results = pd.DataFrame.from_dict(res_dict)

        if verbose:
            print(self.results)

    def save_results(self, save_to_path: str, model_name: str, plot: bool = False):
        """
        Save the final metrics to disk in the CSV format

        :param save_to_path: Path on disk
        :param model_name: The name of the text model that was used in the evaluation
        :param plot: Should draw a plot or not
        """
        if self.results is None:
            raise ValueError("No results to save")

        os.makedirs(save_to_path, exist_ok=True)

        self.results.to_csv(os.path.join(save_to_path, model_name + ".csv"))
        if plot:
            self.__plot_metric(Metric.Acc, model_name, save_to_path)
            self.__plot_metric(Metric.MAP, model_name, save_to_path)

    def __plot_metric(self, metric_name, model_name, save_to_path):
        metric_results = self.results.loc[:, self.results.columns.str.endswith(metric_name)].copy()
        metric_results["k"] = self.results.k

        metric_results.plot(
            x="k",
            kind="line",
            marker="v",
            figsize=(7, 5),
            title=model_name,
            ylabel=metric_name,
            grid=True,
        )
        plt.savefig(os.path.join(save_to_path, model_name + "_" + metric_name + ".png"))

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

    def _load_models(
        self, model_types: List[str], model_class: Type[AbstractModel], models_directory: str
    ) -> Dict[str, AbstractModel]:
        return {
            train_type: model_class.load(os.path.join(models_directory, model_class.name + "_" + train_type))
            for train_type in model_types
        }

    def __get_relevant_reports_num(self) -> Counter:
        result: Counter = Counter()

        test_disc_ids = self.test.disc_id.tolist()
        count_train: Counter = Counter(self.train.disc_id)

        for i, query_id in enumerate(self.test.id):
            result[query_id] += count_train[test_disc_ids[i]]
            count_train[test_disc_ids[i]] += 1

        return result
