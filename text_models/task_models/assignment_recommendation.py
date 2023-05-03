import logging
from typing import List, Union, Iterable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import wandb
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, Tensor
from torch.utils.data import Dataset

from data_processing.util import get_corpus, Corpus
from text_models.evaluation import ListDataset, AssignmentEvaluator
from text_models.task_models import AbstractTask

logger = logging.getLogger(__name__)


class SoftmaxClassifier(nn.Module):
    def __init__(self, model: SentenceTransformer, weights: torch.Tensor = None):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="mean")

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Optional[Tensor] = None):
        output = self.model(sentence_features[0])["sentence_embedding"]  # type: ignore
        if labels is not None:
            return self.loss_fn(output, labels.view(-1))
        return output


class AssignmentRecommendationTask(AbstractTask):
    def __init__(self, corpus: Corpus, labels: List[str], config):
        self.num_labels = len(set(labels))
        self.map_labels = self.numerate_labels(labels)
        super().__init__(corpus, labels, config)
        self.classes_weights = compute_class_weight("balanced", classes=list(self.map_labels.values()),
                                                    y=[self.map_labels[label] for label in labels[:self.train_size]])

    name = "assignment_recommendation"

    def train(
            self,
            word_embedding_model: models.Transformer,
            save_to_dir: str,
            step_metric: Optional[str],
            report_wandb: bool = False,
            hp_search_mode: bool = False,
    ) -> SentenceTransformer:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(self.classes_weights, dtype=torch.float).to(self.config.device))
        lr_finder = self.find_lr(word_embedding_model, criterion, 1e-8, 1e-5, 200)
        min_grad_idx = (np.gradient(np.array(lr_finder.history["loss"]))).argmin()
        self.config.learning_rate = lr_finder.history["lr"][min_grad_idx]
        logger.info(f"lr: {self.config.learning_rate}")
        if report_wandb:
            wandb.log({"lr": self.config.learning_rate})
        lr_finder.reset()
        return super().train(word_embedding_model, save_to_dir, step_metric, report_wandb, hp_search_mode)

    def _create_model(self, word_embedding_model: models.Transformer) -> SentenceTransformer:
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.config.pooling_mode
        )
        dropout = models.Dropout(dropout=self.config.dropout_ratio)
        classifier = models.Dense(word_embedding_model.get_word_embedding_dimension(), self.num_labels,
                                  activation_function=nn.Identity())
        nn.init.xavier_uniform_(classifier.linear.weight)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dropout, classifier], device=self.config.device)

    def _get_loss(self, model: SentenceTransformer):
        loss = SoftmaxClassifier(model, torch.tensor(self.classes_weights, dtype=torch.float))
        self.evaluator.softmax_model = loss
        return loss

    def _get_dataset(self, corpus: List[str], labels: List[str], n_examples: Union[str, int]) -> Dataset:
        return ListDataset(
            [
                InputExample(texts=[bug_description], label=label)
                for bug_description, label in zip(corpus, [self.map_labels[label] for label in labels])
            ]
        )

    def _get_evaluator(
            self, train_corpus: List[str], train_labels: List[str], val_corpus: List[str], val_labels: List[str]
    ) -> SentenceEvaluator:
        return AssignmentEvaluator(val_corpus, [self.map_labels[label] for label in val_labels], self.num_labels, val_corpus, **self.config.evaluator_config)

    @staticmethod
    def numerate_labels(labels: List[str]) -> Dict[str, int]:
        result = {}
        i = 0
        for label in labels:
            if label not in result:
                result[label] = i
                i += 1
        return result

    @classmethod
    def load(cls, data: pd.DataFrame, config):
        corpus = get_corpus(data, sentences=True)
        labels = data["assignee"].tolist()
        return AssignmentRecommendationTask(corpus, labels, config)  # type: ignore
