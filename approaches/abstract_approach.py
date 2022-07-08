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

        self.train["id_num"] = np.arange(len(self.train.index))
        self.test["id_num"] = np.arange(len(self.test.index))

        self.train_corpus: List = get_corpus(train)
        self.test_corpus: List = get_corpus(test)

        self.results: Optional[pd.DataFrame] = None
        self.test_size: Optional[int] = None
        self.true_positive: Optional[np.ndarray] = None

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
                res_dict[name] = self.evaluate(model, topns)

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
            self.results.plot(
                x="k",
                kind="line",
                marker="o",
                figsize=(7, 5),
                title=model_name,
                ylabel="SuccessRate@k",
                grid=True,
            )
            plt.savefig(os.path.join(save_to_path, model_name + ".png"))

    def evaluate(self, model: AbstractModel, topns: List[int]) -> np.ndarray:
        """
        Evaluate model.

        :param model: Evaluation model
        :param topns: What number of the most similar bug reports according to the model will be used in the evaluation
        :return: SuccessRate@n for all topns from topns parameter.
        """
        if model is None:
            return np.zeros(len(topns))

        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)

        self.setup_approach()

        self.test_size = 0
        self.true_positive = np.zeros(len(topns))

        def eval_sample(query_report):
            if query_report.id != query_report.disc_id:  # not in master ids
                dupl_ids = self.get_duplicated_ids(query_report.id_num, max(topns))
                for i, topn in enumerate(topns):
                    self.true_positive[i] += np.any(self.train.iloc[dupl_ids[:topn]]["disc_id"] == query_report.disc_id)
                self.test_size += 1
            self.train = self.train.append(query_report, ignore_index=True)
            self.update_history(query_report.id_num)

        self.test.apply(eval_sample, axis=1)

        return self.true_positive / self.test_size

    @abstractmethod
    def setup_approach(self):
        raise NotImplementedError()

    @abstractmethod
    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_history(self, query_num: int):
        raise NotImplementedError()
