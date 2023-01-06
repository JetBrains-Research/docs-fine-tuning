import os
from abc import abstractmethod
from typing import Union

from sentence_transformers import evaluation, models
from torch.utils.data import Dataset
from transformers import (
    AutoModelForNextSentencePrediction,
    AutoTokenizer,
    TrainingArguments,
    IntervalStrategy,
    AutoConfig,
    PreTrainedTokenizerBase,
)

from data_processing.util import Corpus
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.evaluation import IREvalTrainer, ValMetric


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

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: int = 200,
        n_examples: Union[str, int] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_best_model: bool = False,
        save_steps: int = 2000,
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, val, metric_for_best_model, save_best_model)
        self.save_steps = save_steps

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:

        config = AutoConfig.from_pretrained(pretrained_model)
        model = AutoModelForNextSentencePrediction.from_pretrained(pretrained_model, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        dataset = self._get_dataset(docs_corpus, tokenizer, max_len)
        val_dataset = self._get_dataset(evaluator.corpus, tokenizer, max_len)
        return self._train_and_save(model, tokenizer, dataset, val_dataset, evaluator, save_to_path,
                                    self.save_steps, max_len, device)

    @abstractmethod
    def _get_dataset(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dataset:
        raise NotImplementedError()

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        return models.Transformer(load_from_path)
