import numpy as np
import pandas as pd
from sentence_transformers.readers import InputExample

from approaches.abstract_approach import AbstractApproach
from data_processing.util import get_corpus, fix_random_seed, flatten
from text_models.evaluation import AssignmentEvaluator, ListDataset, AssigneeMetrics
from text_models.task_models import AssignmentRecommendationTask


class AssignmentApproach(AbstractApproach):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(
            train,
            test,
            [
                AssigneeMetrics.ACC,
                AssigneeMetrics.PRECISION,
                AssigneeMetrics.W_PRECISION,
                AssigneeMetrics.RECALL,
                AssigneeMetrics.F1,
                AssigneeMetrics.WAF1,
            ],
        )
        fix_random_seed(42)

        num_labels = len(set(train.assignee.tolist()))

        corpus = [" ".join(doc) for doc in list(map(flatten, get_corpus(test, sentences=True)))]  # type: ignore
        labels_map = AssignmentRecommendationTask.numerate_labels(train.assignee.tolist())
        labels = [labels_map[label] for label in test.assignee.tolist()]
        self.dataset = ListDataset(
            [InputExample(texts=[sentence], label=label) for sentence, label in zip(corpus, labels)]
        )

        self.evaluator = AssignmentEvaluator(self.dataset, num_labels, write_csv=False)

    def evaluate(self, model, topns):
        self.evaluator.acc_at_k = topns
        self.evaluator.f1_at_k = topns
        self.evaluator.recall_at_k = topns
        self.evaluator.precision_at_k = topns
        return {metric_name: np.array(val) for metric_name, val in self.evaluator.compute_metrics(model.model).items()}
