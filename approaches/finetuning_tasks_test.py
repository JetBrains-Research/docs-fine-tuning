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

    def evaluate_all(
        self,
        from_scratch_model: AbstractModel,
        pretrained_model: AbstractModel,
        finetuned_model: AbstractModel,
        topns: List[int],
    ):
        self.approach.evaluate_all(from_scratch_model, pretrained_model, finetuned_model, topns)
        self.results_list = [self.approach.results]

        for i in range(1, len(self.tasks)):
            task_name = self.tasks[i]
            model_finetuned = BertSiameseModel.load(
                os.path.join(self.models_directory, BertSiameseModel.name + "_" + task_name + self.model_suffix)
            )
            res_copy = self.approach.results.copy()
            res_copy["PT+DOC+TASK"] = self.approach.evaluate(model_finetuned, topns)
            print(res_copy)
            self.results_list.append(res_copy)

    def save_results(self, save_to_path, model_name, graph=False):
        for i, result in enumerate(self.results_list):
            self.results = result
            super(FinetuningTasksTest, self).save_results(os.path.join(save_to_path, self.tasks[i]), model_name, graph)

    def setup_approach(self):
        self.approach.setup_approach()

    def update_history(self, query_num: int):
        self.approach.update_history(query_num)

    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        return self.approach.get_duplicated_ids(query_num, topn)
