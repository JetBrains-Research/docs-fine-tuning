import os
from typing import List, Dict, Type

import numpy as np

from approaches import AbstractApproach
from text_models import AbstractModel


class PretrainingTasksTest(AbstractApproach):
    """
    The approach to evaluate all fine-tuning tasks for a transformer-based model (BertSiameseModel).

    :param approach: Base approach
    :param pretraining_tasks: Tasks to evaluate
    """

    def __init__(self, approach: AbstractApproach, pretraining_tasks: List[str]):
        super(PretrainingTasksTest, self).__init__(approach.train, approach.test)
        self.approach = approach
        self.pretraining_tasks = pretraining_tasks

    def _load_models(
        self, model_types: List[str], model_class: Type[AbstractModel], models_directory: str
    ) -> Dict[str, AbstractModel]:
        res = {}
        for train_type in model_types:
            if "DOC" in train_type or "BUG" in train_type:
                for task in self.pretraining_tasks:
                    res[task + "_" + train_type] = model_class.load(
                        os.path.join(models_directory, model_class.name + "_" + task + "_" + train_type)
                    )
            else:
                res[train_type] = model_class.load(os.path.join(models_directory, model_class.name + "_" + train_type))

        return res

    def evaluate(self, model: AbstractModel, topks: List[int]) -> Dict[str, np.ndarray]:
        return self.approach.evaluate(model, topks)

    def setup_approach(self):
        pass

    def get_duplicated_ids(self, query_num: int, topn: int) -> np.ndarray:
        pass

    def update_history(self, query_num: int):
        pass