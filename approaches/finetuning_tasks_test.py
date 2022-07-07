import os
from typing import List

import numpy as np
import pandas as pd

from approaches import AbstractApproach
from text_models import AbstractModel, TrainTypes, BertSiameseModel


class FinetuningTasksTest(AbstractApproach):
    def __init__(
        self,
        approach: AbstractApproach,
        tasks: List[str],
        models_directory: str,
    ):
        super().__init__(approach.train, approach.test)
        self.tasks = tasks
        self.models_directory = models_directory
        self.approach = approach

    def evaluate_all(
        self,
        task_model: AbstractModel,
        pt_task_model: AbstractModel,
        doc_task_model: AbstractModel,
        pt_doc_task_model: AbstractModel,
        topns: List[int],
        verbose: bool = True,
    ):
        self.approach.evaluate_all(task_model, pt_task_model, doc_task_model, pt_doc_task_model, topns, verbose=False)
        self.results_list = [self.approach.results]

        if pt_doc_task_model is None and doc_task_model is None:
            return

        self.all_results: pd.DataFrame = self.approach.results.copy()  # type: ignore
        self.all_results.rename(columns={TrainTypes.PT_DOC_TASK: f"PT_DOC({self.tasks[0]})_TASK"}, inplace=True)
        self.all_results.rename(columns={TrainTypes.DOC_TASK: f"DOC({self.tasks[0]})_TASK"}, inplace=True)

        # we need to skip first task because we have already processed this task
        # with the self.approach.evaluate_all() method call above
        for i in range(1, len(self.tasks)):
            task_name = self.tasks[i]
            res_copy = self.approach.results.copy()  # type: ignore

            if doc_task_model is not None:
                model_doc_task = BertSiameseModel.load(
                    os.path.join(
                        self.models_directory, BertSiameseModel.name + "_" + task_name + "_" + TrainTypes.DOC_TASK
                    )
                )
                task_doc_test_res = self.approach.evaluate(model_doc_task, topns)
                res_copy[TrainTypes.DOC_TASK] = task_doc_test_res
                self.all_results[f"DOC({self.tasks[i]})_TASK"] = task_doc_test_res

            if pt_doc_task_model is not None:
                model_pt_doc_task = BertSiameseModel.load(
                    os.path.join(
                        self.models_directory, BertSiameseModel.name + "_" + task_name + "_" + TrainTypes.PT_DOC_TASK
                    )
                )
                task_result = self.approach.evaluate(model_pt_doc_task, topns)
                res_copy[TrainTypes.PT_DOC_TASK] = task_result
                self.all_results[f"PT_DOC({self.tasks[i]})_TASK"] = task_result

            self.results_list.append(res_copy)

        if verbose:
            print(self.all_results)

    def save_results(self, save_to_path: str, model_name: str, plot: bool = False):
        for i, result in enumerate(self.results_list):
            self.results = result
            super().save_results(save_to_path, model_name + "_" + self.tasks[i], plot)

        if self.all_results is not None:
            self.results = self.all_results
            super().save_results(save_to_path, model_name, plot)

    def setup_approach(self):
        self.approach.setup_approach()

    def update_history(self, query_num: int):
        self.approach.update_history(query_num)

    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        return self.approach.get_duplicated_ids(query_num, topn)
