import os
from abc import ABC, abstractmethod
from typing import List, Type, Dict, Union, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from text_models import AbstractModel


class AbstractApproach(ABC):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, metrics: List[str]):
        self.train = train
        self.test = test
        self.metrics = metrics

        self.results: Optional[pd.DataFrame] = None

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

    @abstractmethod
    def evaluate(self, model: AbstractModel, topns: List[int]):
        raise NotImplementedError()

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
            for metric in self.metrics:
                self.__plot_metric(metric, model_name, save_to_path)

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

    def _load_models(
        self, model_types: List[str], model_class: Type[AbstractModel], models_directory: str
    ) -> Dict[str, AbstractModel]:
        return {
            train_type: model_class.load(os.path.join(models_directory, model_class.name + "_" + train_type))
            for train_type in model_types
        }
