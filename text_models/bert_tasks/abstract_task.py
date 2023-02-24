import os
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional

from datasets import Dataset as HFDataset
from sentence_transformers import models, evaluation
from torch.utils.data import random_split, Dataset
from transformers import IntervalStrategy, TrainingArguments, PreTrainedTokenizer, PreTrainedModel, DataCollator

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
        eval_steps: Optional[int] = None,  # if None then epoch mode will be used
        n_examples: Union[str, int] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_steps: Optional[int] = None,  # if None then epoch mode will be used
        save_best_model: bool = False,
        do_eval_on_artefacts: bool = True,
        max_len: Optional[int] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.eval_steps = eval_steps
        self.save_best_model = save_best_model
        self.val = val
        self.metric_for_best_model = metric_for_best_model
        self.save_steps = save_steps
        self.do_eval_on_artefacts = do_eval_on_artefacts
        self.max_len = max_len

    @abstractmethod
    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,  # list of list(sections) of list(sentences) of tokens(words)
        evaluator: evaluation.InformationRetrievalEvaluator,
        device: str,
        save_to_path: str,
        report_wandb: bool = False,
    ) -> models.Transformer:
        """
        Load, fine-tune and save model.

        :param pretrained_model: Path on disk or name of pre-trained model
        :param docs_corpus: Corpus of documentation sections
        :param evaluator: The Information Retrieval Evaluator for validation
        :param device: What device will be used for training. Possible values: "cpu", "cuda".
        :param save_to_path: Where the fine-tuned model should be saved
        :param report_wandb: Whether report to wandb or not
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

    def _train_and_save(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Union[Dataset, HFDataset],
        val_task_dataset: Union[Dataset, HFDataset],
        evaluator: evaluation.InformationRetrievalEvaluator,
        save_to_path: str,
        save_steps: Optional[int],
        device: str,
        report_wandb: bool = False,
        data_collator: Optional[DataCollator] = None,
    ) -> models.Transformer:

        train_dataset, val_dataset = self._train_val_split(dataset)
        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")

        args = TrainingArguments(
            output_dir=checkpoints_path,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=evaluator.batch_size,
            num_train_epochs=self.epochs,
            save_strategy=IntervalStrategy.EPOCH if self.save_steps is None else IntervalStrategy.STEPS,
            save_steps=save_steps,  # type: ignore
            evaluation_strategy=IntervalStrategy.EPOCH if self.eval_steps is None else IntervalStrategy.STEPS,
            eval_steps=self.eval_steps,  # type: ignore
            load_best_model_at_end=self.save_best_model,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=(self.metric_for_best_model == "task_map"),
            disable_tqdm=False,
            do_eval=True,
            report_to="none",  # type: ignore
        )

        trainer = IREvalTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,  # type: ignore
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        trainer.set_env_vars(
            evaluator, model.bert, tokenizer, val_task_dataset, max_len, self.name, device, report_wandb  # type: ignore
        )
        trainer.train()
        # if self.save_best_model == True we will use best model
        output_path = os.path.join(save_to_path, "output_docs")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return models.Transformer(output_path)

    def _train_val_split(
        self, dataset: Union[Dataset, HFDataset]
    ) -> Tuple[Union[Dataset, HFDataset], Optional[Dataset]]:
        if not self.do_eval_on_artefacts:
            return dataset, None
        train_size = int((1 - self.val) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])  # type: ignore
        return train_dataset, eval_dataset
