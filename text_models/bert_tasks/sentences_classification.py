import os
from abc import abstractmethod
from typing import Union, Optional

from sentence_transformers import evaluation, models
from torch.utils.data import Dataset
from transformers import (
    AutoModelForNextSentencePrediction,
    AutoTokenizer,
    AutoConfig,
    PreTrainedTokenizerBase,
)

from data_processing.util import Corpus
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.evaluation import ValMetric


class SentencesClassificationTask(AbstractTask):
    """
    Base class for all sentence classification tasks.

    :param epochs: Number of fine-tuning epochs
    :param batch_size: Batch size used for fine-tuning
    :param eval_steps: Number of update steps between two evaluations
    :param n_examples: Number of input examples that will be used for fine-tuning
    :param save_best_model: Whether or not to save the best model found during training at the end of training
    :param save_steps: Number of updates steps before two checkpoint saves
    """

    name = "abstract_sentence_classification"

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
        report_wandb: bool = False,
    ) -> models.Transformer:

        config = AutoConfig.from_pretrained(pretrained_model)
        model = AutoModelForNextSentencePrediction.from_pretrained(pretrained_model, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        dataset = self._get_dataset(docs_corpus, tokenizer, max_len)
        val_dataset = self._get_dataset(evaluator.val_dataset, tokenizer, max_len)  # type: ignore
        return self._train_and_save(
            model,
            tokenizer,
            dataset,
            val_dataset,
            evaluator,
            save_to_path,
            self.save_steps,
            max_len,
            device,
            report_wandb,
        )

    @abstractmethod
    def _get_dataset(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dataset:
        raise NotImplementedError()

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        return models.Transformer(load_from_path)
