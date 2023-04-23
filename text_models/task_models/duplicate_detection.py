from typing import List

import pandas as pd
from sentence_transformers import models, SentenceTransformer, losses
from sentence_transformers.evaluation import SentenceEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from torch import nn
from torch.utils.data import Dataset

from data_processing.util import get_corpus
from text_models.datasets import CosineSimilarityDataset, TripletDataset
from text_models.task_models import AbstractTask


class DuplicatesDetection(AbstractTask):
    name = "duplicates_detection"

    def _create_model(self, word_embedding_model) -> SentenceTransformer:
        word_embedding_model.max_seq_length = self.config.max_len
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.config.pooling_mode
        )
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.config.vector_size,
            activation_function=nn.Tanh(),
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                   device=self.config.device)

    def _get_loss(self, model: SentenceTransformer):
        return (
            losses.CosineSimilarityLoss(model)
            if self.config.loss == "cossim"
            else losses.TripletLoss(
                model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=1
            )
        )

    def _get_evaluator(
            self, train_corpus: List[str], train_labels: List[str], val_corpus: List[str], val_labels: List[str]
    ) -> SentenceEvaluator:
        queries = {qid: query for qid, query in enumerate(val_corpus)}
        corpus = {cid: doc for cid, doc in enumerate(train_corpus)}
        relevant_docs = {
            qid: {cid for cid in corpus.keys() if train_labels[cid] == val_labels[qid]} for qid in queries.keys()
        }

        evaluator = InformationRetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            main_score_function="cos_sim",
            score_functions={"cos_sim": cos_sim},  # type: ignore
            **self.config.evaluator_config,
        )
        evaluator.metrics = None
        evaluator.val_dataset = val_corpus
        return evaluator

    def _get_dataset(self, corpus, disc_ids, n_examples) -> Dataset:
        if self.config.loss == "cossim":
            return CosineSimilarityDataset(corpus, disc_ids, n_examples, shuffle=True)
        if self.config.loss == "triplet":
            return TripletDataset(corpus, disc_ids, n_examples, shuffle=True)
        raise ValueError("Unsupported loss")

    @classmethod
    def load(cls, data: pd.DataFrame, config):
        corpus = get_corpus(data, sentences=True)
        labels = data["disc_ids"].tolist()
        return DuplicatesDetection(corpus, labels, config)
