import os
from abc import ABC

import numpy as np
import pandas as pd

from text_models import AbstractModel
from data_processing.util import get_corpus
from typing import List
import matplotlib.pyplot as plt


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
        finetuned_model: AbstractModel,
        topns: List[int],
    ):
        res_dict = {"topn": topns, "TASK": [], "PT+TASK": [], "PT+DOC+TASK": []}
        for topn in topns:
            from_scratch = self.evaluate(from_scratch_model, topn)
            pretrained = self.evaluate(pretrained_model, topn)
            finetuned = self.evaluate(finetuned_model, topn)
            res_dict["TASK"].append(from_scratch)
            res_dict["PT+TASK"].append(pretrained)
            res_dict["PT+DOC+TASK"].append(finetuned)
        self.results = pd.DataFrame.from_dict(res_dict)
        print(self.results)

    def save_results(self, save_to_path, model_name, graph=False):
        self.results.to_csv(os.path.join(save_to_path, model_name + ".csv"))
        if graph:
            self.results.plot(
                x="topn",
                y=["TASK", "PT+TASK", "PT+DOC+TASK"],
                kind="line",
                marker="o",
                figsize=(7, 5),
                title="Random",
                ylabel="success rate",
                grid=True,
            )
            plt.savefig(os.path.join(save_to_path, model_name + ".png"))

    def evaluate(self, model: AbstractModel, topn=5):
        self.embeddings = model.get_embeddings(self.train_corpus)
        self.test_embs = model.get_embeddings(self.test_corpus)

        self.setup_approach()

        self.test_size = 0
        self.TP = 0

        def __eval_sample(query_report):
            if query_report.id != query_report.disc_id:  # not in master ids
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
