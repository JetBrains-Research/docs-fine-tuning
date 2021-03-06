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
from text_models.bert_tasks.eval_trainer import IREvalTrainer


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
        save_best_model: bool = False,
        save_steps: int = 2000,
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, save_best_model)
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

        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")
        output_path = os.path.join(save_to_path, "output_docs")
        args = TrainingArguments(
            output_dir=checkpoints_path,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=self.save_steps,
            save_total_limit=3,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.eval_steps,
            load_best_model_at_end=self.save_best_model,
            metric_for_best_model=f"MAP@{max(evaluator.map_at_k)}",
            greater_is_better=True,
            disable_tqdm=False,
        )

        trainer = IREvalTrainer(model=model, args=args, train_dataset=dataset)
        trainer.set_env_vars(evaluator, model.bert, tokenizer, self.name, max_len, device)
        trainer.train()

        # if self.save_best_model == True we will use best model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return models.Transformer(output_path)

    @abstractmethod
    def _get_dataset(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dataset:
        raise NotImplementedError()

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        return models.Transformer(load_from_path)
