import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.util import get_corpus
from text_models import AbstractModel, TrainTypes


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
        task_model: AbstractModel,
        pt_task_model: AbstractModel,
        doc_task_model: AbstractModel,
        pt_doc_task_model: AbstractModel,
        topns: List[int],
        verbose: bool = True,
    ):
        """
        Evaluate all models trained in different ways.

        :param task_model: Model trained from scratch on the final task
        :param pt_task_model: Model trained on the final task using pre-trained model
        :param doc_task_model: Model trained from scratch on docs and then train on the final task
        :param pt_doc_task_model: Model trained on docs using a pre-trained model and then train on the task
        :param topns: What number of the most similar bug reports according to the model will be used in the evaluation
        :param verbose: Should evaluation logs be verbose or not
        """

        res_dict: Dict[str, Union[np.ndarray, List[int]]] = {"k": topns}
        models_dict = {
            TrainTypes.TASK: task_model,
            TrainTypes.PT_TASK: pt_task_model,
            TrainTypes.DOC_TASK: doc_task_model,
            TrainTypes.PT_DOC_TASK: pt_doc_task_model,
        }
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
            self.__plot_metric("SuccessRate@k", model_name, save_to_path)
            self.__plot_metric("MAP@k", model_name, save_to_path)

    def __plot_metric(self, metric_name, model_name, save_to_path):
        metric_results = self.results.loc[:, self.results.columns.str.endswith(metric_name)].copy()
        metric_results["k"] = self.results.k

        metric_results.plot(
            x="k",
            kind="line",
            marker="o",
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
        :return: SuccessRate@n for all topns from topns parameter.
        """
        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)
        self.train_copy = self.train.copy()

        self.setup_approach()

        self.test_size = 0
        self.true_positive_at_k = np.zeros(len(topks))
        self.ave_precision_at_k = np.zeros(len(topks))

        def eval_sample(query_report):
            if query_report.id != query_report.disc_id:  # not in master ids
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

                    ave_precision = sum_precisions / min(topk, self.relevant_reports_num[query_report.id])
                    self.ave_precision_at_k[i] += ave_precision

                self.test_size += 1
            self.train_copy = self.train_copy.append(query_report, ignore_index=True)
            self.update_history(query_report.id_num)

        self.test.apply(eval_sample, axis=1)

        metrics = {
            "SuccessRate@k": self.true_positive_at_k / self.test_size,
            "MAP@k": self.ave_precision_at_k / self.test_size,
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

    def __get_relevant_reports_num(self) -> Dict[str, int]:
        result: Dict[str, int] = dict()

        query_ids = self.test.id.tolist()
        train_disc_ids = self.train.disc_id.tolist()
        test_disc_ids = self.test.disc_id.tolist()

        for i, query_id in enumerate(query_ids):
            for train_disc_id in train_disc_ids:
                if train_disc_id == test_disc_ids[i]:
                    if query_id in result.keys():
                        result[query_id] += 1
                    else:
                        result[query_id] = 1
            train_disc_ids.append(test_disc_ids[i])

        return result
