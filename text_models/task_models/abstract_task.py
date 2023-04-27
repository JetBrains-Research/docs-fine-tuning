import os
from abc import ABC, abstractmethod
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from sentence_transformers import models, SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import Dataset, DataLoader

from text_models.evaluation import LossEvaluator
from data_processing.util import Corpus, fix_random_seed, flatten


class AbstractTask(ABC):

    def __init__(self, corpus: Corpus, labels: List[str], config):
        self.save_best_model = None
        fix_random_seed(config.seed)

        self.config = config

        bug_corpus = [" ".join(doc) for doc in list(map(flatten, corpus))]

        self.train_size = int(len(corpus) * (1 - config.val_size))

        train_corpus = bug_corpus[:self.train_size]
        train_disc_ids = labels[:self.train_size]
        val_corpus = bug_corpus[self.train_size:]
        val_disc_ids = labels[self.train_size:]

        self.tapt_data = corpus[:self.train_size]

        self.dataset = self._get_dataset(train_corpus, train_disc_ids, config.n_examples)
        self.eval_dataset = self._get_dataset(val_corpus, val_disc_ids, "all")
        self.evaluator = self._get_evaluator(train_corpus, train_disc_ids, val_corpus, val_disc_ids)

    def train(self, word_embedding_model: models.Transformer, save_to_dir: str,
              step_metric: Optional[str], report_wandb: bool = False,
              hp_search_mode: bool = False) -> SentenceTransformer:
        model = self._create_model(word_embedding_model)

        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=self.config.batch_size)
        train_loss = self._get_loss(model)

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
            scheduler=self.config.scheduler,
            evaluator=evaluator,
            evaluation_steps=0 if self.config.evaluation_steps is None else self.config.evaluation_steps,
            output_path=None if hp_search_mode else output_path,
            checkpoint_path=None if hp_search_mode else checkpoint_path,
            show_progress_bar=True,
            checkpoint_save_steps=None if hp_search_mode else checkpoint_save_steps,
            save_best_model=self.config.save_best_model,
        )

        return model

    @abstractmethod
    def _create_model(self, word_embedding_model: models.Transformer) -> SentenceTransformer:
        raise NotImplementedError()

    @abstractmethod
    def _get_loss(self, model: SentenceTransformer):
        raise NotImplementedError()

    @abstractmethod
    def _get_evaluator(self, train_corpus: List[str], train_labels: List[str], val_corpus: List[str],
                       val_labels: List[str]) -> SentenceEvaluator:
        raise NotImplementedError()

    @abstractmethod
    def _get_dataset(self, corpus: List[str], labels: List[str], n_examples: Union[str, int]) -> Dataset:
        raise NotImplementedError()

    @classmethod
    def load(cls, data: pd.DataFrame, config):
        raise NotImplementedError()
