from typing import List, Union, Iterable, Dict

import pandas as pd
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor

from data_processing.util import get_corpus, Corpus
from text_models.evaluation import ListDataset, AccuracyEvaluator
from text_models.task_models import AbstractTask


class SoftmaxClassifier(nn.Module):
    def __init__(self, model: SentenceTransformer, sentence_embedding_dimension: int, num_labels: int):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(sentence_embedding_dimension, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        embeddings = self.model(sentence_features[0])['sentence_embedding']
        output = self.classifier(embeddings)
        if labels is not None:
            return self.loss_fn(output, labels.view(-1))
        return embeddings, output


class AssignmentRecommendationTask(AbstractTask):

    def __init__(self, corpus: Corpus, labels: List[str], config):
        super().__init__(corpus, labels, config)
        self.num_labels = len(set(labels))

    name = "assignment_recommendation"

    def _create_model(self, word_embedding_model: models.Transformer) -> SentenceTransformer:
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.config.pooling_mode
        )
        dropout = models.Dropout(dropout=self.config.dropout_ratio)
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dropout],
                                   device=self.config.device)

    def _get_loss(self, model: SentenceTransformer):
        return SoftmaxClassifier(model, model.get_sentence_embedding_dimension(), self.num_labels)

    def _get_dataset(self, corpus: List[str], labels: List[str], n_examples: Union[str, int]) -> Dataset:
        return ListDataset([InputExample(texts=[bug_description], label=label) for bug_description, label in
                            zip(corpus, self.__numerate_labels(labels))])

    def _get_evaluator(self, train_corpus: List[str], train_labels: List[str], val_corpus: List[str],
                       val_labels: List[str]) -> SentenceEvaluator:
        eval_dataloader = DataLoader(self.eval_dataset, shuffle=True, **self.config.evaluator_config)
        return AccuracyEvaluator(eval_dataloader)

    def __numerate_labels(self, labels: List[str]) -> List[int]:
        d = {label: i for i, label in enumerate(set(labels))}
        return [d[label] for label in labels]

    @classmethod
    def load(cls, data: pd.DataFrame, config):
        corpus = get_corpus(data, sentences=True)
        labels = data["assignee"].tolist()
        return AssignmentRecommendationTask(corpus, labels, config)
