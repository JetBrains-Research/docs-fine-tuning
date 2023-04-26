from typing import List, Type

import numpy as np
import pandas as pd
from sentence_transformers.readers import InputExample

from approaches import AbstractApproach
from data_processing.util import get_corpus, fix_random_seed, flatten
from text_models import AbstractModel
from text_models.evaluation import AssignmentEvaluator, ListDataset, AssigneeMetrics
from text_models.task_models import AssignmentRecommendationTask


class AssignmentApproach(AbstractApproach):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test, [AssigneeMetrics.ACC, AssigneeMetrics.WAF1, AssigneeMetrics.WP, AssigneeMetrics.WR])
        fix_random_seed(42)

        num_labels = len(set(train.assignee.tolist() + test.assignee.tolist()))

        corpus = [" ".join(doc) for doc in list(map(flatten, get_corpus(test, sentences=True)))]
        labels = AssignmentRecommendationTask.numerate_labels(test["assignee"].tolist())
        self.dataset = ListDataset(
            [InputExample(texts=[sentence], label=label) for sentence, label in zip(corpus, labels)])

        self.evaluator = AssignmentEvaluator(self.dataset, num_labels, write_csv=False)

    def evaluate_all(
        self,
        model_types: List[str],
        model_class: Type[AbstractModel],
        models_directory: str,
        topns: List[int],
        verbose: bool = True,
    ):
        super().evaluate_all(model_types, model_class, models_directory, [1], verbose)

    def evaluate(self, model, topns):
        return {metric_name: np.array(val) for metric_name, val in self.evaluator.compute_metrics(model.model).items()}
