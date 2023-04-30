from typing import List, Union, Iterable, Dict

import pandas as pd
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.readers import InputExample
from torch import nn, Tensor
from torch.utils.data import Dataset

from data_processing.util import get_corpus, Corpus
from text_models.evaluation import ListDataset, AssignmentEvaluator
from text_models.task_models import AbstractTask


class SoftmaxClassifier(nn.Module):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = self.model(sentence_features[0])["sentence_embedding"]  # type: ignore
        output = self.classifier(embeddings)
        if labels is not None:
            return self.loss_fn(output, labels.view(-1))
        return embeddings, output


class AssignmentRecommendationTask(AbstractTask):
    def __init__(self, corpus: Corpus, labels: List[str], config):
        self.num_labels = len(set(labels))
        super().__init__(corpus, labels, config)

    name = "assignment_recommendation"

    def _create_model(self, word_embedding_model: models.Transformer) -> SentenceTransformer:
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.config.pooling_mode
        )
        dropout = models.Dropout(dropout=self.config.dropout_ratio)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dropout], device=self.config.device)

    def _get_loss(self, model: SentenceTransformer):
        loss = SoftmaxClassifier(model, model.get_sentence_embedding_dimension(), self.num_labels)
        self.evaluator.softmax_model = loss
        return loss

    def _get_dataset(self, corpus: List[str], labels: List[str], n_examples: Union[str, int]) -> Dataset:
        return ListDataset(
            [
                InputExample(texts=[bug_description], label=label)
                for bug_description, label in zip(corpus, AssignmentRecommendationTask.numerate_labels(labels))
            ]
        )

    def _get_evaluator(
        self, train_corpus: List[str], train_labels: List[str], val_corpus: List[str], val_labels: List[str]
    ) -> SentenceEvaluator:
        return AssignmentEvaluator(self.eval_dataset, self.num_labels, val_corpus, **self.config.evaluator_config)

    @staticmethod
    def numerate_labels(labels: List[str]) -> List[int]:
        d = {label: i for i, label in enumerate(set(labels))}
        return [d[label] for label in labels]

    @classmethod
    def load(cls, data: pd.DataFrame, config):
        corpus = get_corpus(data, sentences=True)
        labels = data["assignee"].tolist()
        return AssignmentRecommendationTask(corpus, labels, config)  # type: ignore
