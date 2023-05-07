import os
from typing import List, Dict, Type

import numpy as np

from approaches.abstract_approach import AbstractApproach
from text_models import AbstractModel, TrainTypes


class PretrainingTasksTest(AbstractApproach):
    """
    The approach to evaluate all fine-tuning tasks for a transformer-based model (BertDomainModel).

    :param approach: Base approach
    :param pretraining_tasks: Tasks to evaluate
    """

    def __init__(self, approach: AbstractApproach, pretraining_tasks: List[str]):
        super(PretrainingTasksTest, self).__init__(approach.train, approach.test, approach.metrics)
        self.approach = approach
        self.pretraining_tasks = pretraining_tasks

    def _load_models(
        self, model_types: List[str], model_class: Type[AbstractModel], models_directory: str
    ) -> Dict[str, AbstractModel]:
        res = {}
        for train_type in model_types:
            if "DOC" in train_type or "BUG" in train_type:
                pretraining_tasks = (
                    self.pretraining_tasks
                    if train_type not in [TrainTypes.DOC_TASK, TrainTypes.BUGS_TASK, TrainTypes.DOC_BUGS_TASK]
                    else ["mlm"]
                )
                for task in pretraining_tasks:
                    res[task + "_" + train_type] = model_class.load(
                        os.path.join(models_directory, model_class.name + "_" + task + "_" + train_type)
                    )
            else:
                res[train_type] = model_class.load(os.path.join(models_directory, model_class.name + "_" + train_type))

        return res

    def evaluate(self, model: AbstractModel, topks: List[int]) -> Dict[str, np.ndarray]:
        return self.approach.evaluate(model, topks)
