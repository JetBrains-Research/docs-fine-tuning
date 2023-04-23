import numpy as np
import pandas as pd
from sentence_transformers.readers import InputExample

from approaches import AbstractApproach
from data_processing.util import get_corpus, fix_random_seed, flatten
from text_models.evaluation import AccuracyEvaluator, ListDataset
from text_models.task_models import AssignmentRecommendationTask


class AssignmentApproach(AbstractApproach):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test, ["Acc"])
        fix_random_seed(42)

        corpus = [" ".join(doc) for doc in list(map(flatten, get_corpus(test, sentences=True)))]
        labels = AssignmentRecommendationTask.numerate_labels(test["assignee"].tolist())
        self.dataset = ListDataset(
            [InputExample(texts=[sentence], label=label) for sentence, label in zip(corpus, labels)])

        self.evaluator = AccuracyEvaluator(self.dataset, write_csv=False)

    def evaluate(self, model, topns):
        return {"Acc": np.array(self.evaluator(model))}
