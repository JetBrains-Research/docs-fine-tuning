import os
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from sentence_transformers import models, evaluation
from torch.utils.data import random_split, Dataset
from transformers import IntervalStrategy, TrainingArguments, PreTrainedTokenizer, PreTrainedModel

from data_processing.util import Corpus
from text_models.bert_tasks.evaluation import IREvalTrainer, ValMetric


class AbstractTask(ABC):
    """
    Base class for all fine-tuning tasks.

    :param epochs: Number of fine-tuning epochs
    :param batch_size: Batch size used for fine-tuning
    :param eval_steps: Number of update steps between two evaluations
    :param n_examples: Number of input examples that will be used for fine-tuning
    :param save_best_model: Whether to save the best model found during training at the end of training.
    """

    name = "abstract"

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: int = 200,
        n_examples: Union[str, int] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_best_model: bool = False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.eval_steps = eval_steps
        self.save_best_model = save_best_model
        self.val = val
        self.metric_for_best_model = metric_for_best_model

    @abstractmethod
    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,  # list of list(sections) of list(sentences) of tokens(words)
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:
        """
        Load, fine-tune and save model.

        :param pretrained_model: Path on disk or name of pre-trained model
        :param docs_corpus: Corpus of documentation sections
        :param evaluator: The Information Retrieval Evaluator for validation
        :param max_len: max_length parameter of tokenizer
        :param device: What device will be used for training. Possible values: "cpu", "cuda".
        :param save_to_path: Where the fine-tuned model should be saved
        :return: fine-tuned transformer-based model
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, load_from_path) -> models.Transformer:
        """
        Load fine-tuned model.

        :param load_from_path: Path on disk
        :return: fine-tuned transformer-based model
        """
        raise NotImplementedError()

    def _train_and_save(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, dataset: Dataset,
               val_task_dataset: Dataset,
               evaluator: evaluation.InformationRetrievalEvaluator, save_to_path: str,
               save_steps: int, max_len: int, device: str) -> models.Transformer:

        train_dataset, val_dataset = self._train_val_split(dataset)
        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")

        args = TrainingArguments(
            output_dir=checkpoints_path,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=evaluator.batch_size,
            num_train_epochs=self.epochs,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=save_steps,
            save_total_limit=3,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.eval_steps,
            load_best_model_at_end=self.save_best_model,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=(self.metric_for_best_model == "task_map"),
            disable_tqdm=False,
            do_eval=True
        )

        trainer = IREvalTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset)
        trainer.set_env_vars(evaluator, model.bert, tokenizer, val_task_dataset, max_len, device)
        trainer.train()
        # if self.save_best_model == True we will use best model
        output_path = os.path.join(save_to_path, "output_docs")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return models.Transformer(output_path)

    def _train_val_split(self, dataset: Dataset) -> Tuple[
        Dataset, Optional[Dataset]]:
        train_size = int((1 - self.val) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, eval_dataset

