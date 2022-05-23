import os
from typing import List

import numpy as np

from approaches import AbstractApproach
from text_models import AbstractModel, BertSiameseModel


class FinetuningTasksTest(AbstractApproach):
    def __init__(
        self,
        approach: AbstractApproach,
        tasks: List[str],
        models_directory: str,
        model_suffix: str,
    ):
        super().__init__(approach.train, approach.test)
        self.tasks = tasks
        self.models_directory = models_directory
        self.model_suffix = model_suffix
        self.approach = approach

        self.all_results = None
        self.results_list = None

    def evaluate_all(
        self,
        from_scratch_model: AbstractModel,
        pretrained_model: AbstractModel,
        doc_task_model: AbstractModel,
        finetuned_model: AbstractModel,
        topns: List[int],
        silence: bool = False,
    ):
        self.approach.evaluate_all(
            from_scratch_model, pretrained_model, doc_task_model, finetuned_model, topns, silence=True
        )
        self.results_list = [self.approach.results]

        if finetuned_model is None and doc_task_model is None:
            return

        self.all_results = self.approach.results.copy()
        self.all_results.rename(columns={AbstractModel.finetuned: f"PT+DOC({self.tasks[0]})+TASK"}, inplace=True)

        for i in range(1, len(self.tasks)):
            task_name = self.tasks[i]
            res_copy = self.approach.results.copy()

            if finetuned_model is not None:
                model_finetuned = BertSiameseModel.load(
                    os.path.join(self.models_directory, BertSiameseModel.name + "_" + task_name + self.model_suffix)
                )
                task_result = self.approach.evaluate(model_finetuned, topns)
                res_copy[AbstractModel.finetuned] = task_result
                self.all_results[f"PT+DOC({self.tasks[i]})+TASK"] = task_result
                self.results_list.append(res_copy)

            if doc_task_model is not None:
                model_doc_task = BertSiameseModel.load(
                    os.path.join(self.models_directory, BertSiameseModel.name + "_" + task_name + self.model_suffix)
                )
                task_doc_test_res = self.approach.evaluate(model_doc_task, topns)
                res_copy[AbstractModel.doc_task] = task_doc_test_res
                self.all_results[f"DOC({self.tasks[i]})+TASK"] = task_doc_test_res
                self.results_list.append(res_copy)

        print(self.all_results)

    def save_results(self, save_to_path, model_name, graph=False):
        for i, result in enumerate(self.results_list):
            self.results = result
            super(FinetuningTasksTest, self).save_results(save_to_path, model_name + "_" + self.tasks[i], graph)

        if self.all_results is not None:
            self.results = self.all_results
            super(FinetuningTasksTest, self).save_results(save_to_path, model_name, graph)

    def setup_approach(self):
        self.approach.setup_approach()

    def update_history(self, query_num: int):
        self.approach.update_history(query_num)

    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        return self.approach.get_duplicated_ids(query_num, topn)
