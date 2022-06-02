import os
from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.util import get_corpus
from text_models import AbstractModel, TrainTypes


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
        self.results = None
        self.test_size = None
        self.TP = None

    def evaluate_all(
        self,
        from_scratch_model: AbstractModel,
        pretrained_model: AbstractModel,
        doc_task_model: AbstractModel,
        finetuned_model: AbstractModel,
        topns: List[int],
        silence: bool = False,
    ):
        res_dict = {"topn": topns}
        models_dict = {
            TrainTypes.TASK: from_scratch_model,
            TrainTypes.PT_TASK: pretrained_model,
            TrainTypes.DOC_TASK: doc_task_model,
            TrainTypes.PT_DOC_TASK: finetuned_model,
        }
        for name, model in models_dict:
            if model is not None:
                res_dict[name] = self.evaluate(model, topns)

        self.results = pd.DataFrame.from_dict(res_dict)

        if not silence:
            print(self.results)

    def save_results(self, save_to_path, model_name, graph=False):
        if self.results is None:
            raise ValueError("No results to save")

        os.makedirs(save_to_path, exist_ok=True)

        self.results.to_csv(os.path.join(save_to_path, model_name + ".csv"))
        if graph:
            self.results.plot(
                x="topn",
                kind="line",
                marker="o",
                figsize=(7, 5),
                title=model_name,
                ylabel="success rate",
                grid=True,
            )
            plt.savefig(os.path.join(save_to_path, model_name + ".png"))

    def evaluate(self, model: AbstractModel, topns: List[int]) -> np.ndarray:
        if model is None:
            return np.zeros(len(topns))

        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)

        self.setup_approach()

        self.test_size = 0
        self.TP = np.zeros(len(topns))

        def eval_sample(query_report):
            if query_report.id != query_report.disc_id:  # not in master ids
                dupl_ids = self.get_duplicated_ids(query_report.id_num, max(topns))
                for i, topn in enumerate(topns):
                    self.TP[i] += np.any(self.train.iloc[dupl_ids[:topn]]["disc_id"] == query_report.disc_id)
                self.test_size += 1
            self.train = self.train.append(query_report, ignore_index=True)
            self.update_history(query_report.id_num)

        self.test.apply(eval_sample, axis=1)

        self.embeddings = None
        self.test_embs = None

        return self.TP / self.test_size

    @abstractmethod
    def setup_approach(self):
        raise NotImplementedError()

    @abstractmethod
    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update_history(self, query_num: int):
        raise NotImplementedError()
