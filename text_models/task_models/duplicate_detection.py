import os
from typing import Optional, List

import numpy as np
from sentence_transformers import models, SentenceTransformer, losses
from sentence_transformers.evaluation import SentenceEvaluator, InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from torch import nn
from torch.utils.data import Dataset, DataLoader

from text_models.dapt_tasks.evaluation import LossEvaluator
from text_models.datasets import CosineSimilarityDataset, TripletDataset
from text_models.task_models import AbstractTask


class DuplicatesDetection(AbstractTask):

    name = "duplicates_detection"

    def __init__(self, corpus: List[str], labels: List[str], config):
        super().__init__(config)

        train_size = int(len(corpus) * (1 - config.val_size))
        train_corpus = corpus[:train_size]
        train_disc_ids = labels[:train_size]

        val_corpus = corpus[train_size:]
        val_disc_ids = labels[train_size:]

        self.evaluator = self.__get_evaluator(train_corpus, train_disc_ids, val_corpus, val_disc_ids)
        self.dataset = self.__get_dataset(train_corpus, train_disc_ids, config.n_examples)
        self.eval_dataset = self.__get_dataset(val_corpus, val_disc_ids, "all")

    def train(self, word_embedding_model, save_to_dir: str, step_metric: Optional[str],
              report_wandb: bool = False, hp_search_mode: bool = False) -> SentenceTransformer:

        model = self.__create_model(word_embedding_model)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=self.config.batch_size)
        train_loss = (
            losses.CosineSimilarityLoss(model)
            if self.config.loss == "cossim"
            else losses.TripletLoss(
                model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=1
            )
        )

        evaluator = LossEvaluator(
            self.evaluator, train_loss, None, self.eval_dataset, batch_size=self.config.evaluator_config.batch_size
        )

        evaluator = (
            WandbLoggingEvaluator(evaluator, step_metric, len(train_dataloader))  # type: ignore
            if report_wandb and step_metric is not None
            else evaluator
        )

        output_path = save_to_dir if self.config.save_best_model else os.path.join(save_to_dir, "output")
        checkpoint_path = os.path.join(save_to_dir, "checkpoints")
        checkpoint_save_steps = len(train_dataloader) if self.config.save_steps is None else self.config.save_steps

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.epochs,
            warmup_steps=np.ceil(len(train_dataloader) * self.config.epochs * self.config.warmup_ratio),
            weight_decay=self.config.weight_decay,
            optimizer_params={"lr": self.config.learning_rate},
            evaluator=evaluator,
            evaluation_steps=0 if self.config.evaluation_steps is None else self.config.evaluation_steps,
            output_path=None if hp_search_mode else output_path,
            checkpoint_path=None if hp_search_mode else checkpoint_path,
            show_progress_bar=True,
            checkpoint_save_steps=None if hp_search_mode else checkpoint_save_steps,
            save_best_model=self.config.save_best_model,
        )

        return model

    def __create_model(self, word_embedding_model) -> SentenceTransformer:
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

    def __get_evaluator(
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

    def __get_dataset(self, corpus, disc_ids, n_examples) -> Dataset:
        if self.config.loss == "cossim":
            return CosineSimilarityDataset(corpus, disc_ids, n_examples, shuffle=True)
        if self.config.loss == "triplet":
            return TripletDataset(corpus, disc_ids, n_examples, shuffle=True)
        raise ValueError("Unsupported loss")
