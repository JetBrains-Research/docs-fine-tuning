from abc import ABC, abstractmethod
from typing import Union, Tuple

from sentence_transformers import models, evaluation
from torch.utils.data import random_split, Dataset

from data_processing.util import Corpus


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
        eval_with_task: bool = False,
        save_best_model: bool = False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.eval_steps = eval_steps
        self.save_best_model = save_best_model
        self.val = val
        self.eval_with_task = eval_with_task

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

    def _train_val_split(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        train_size = int((1 - self.val)  * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset =  random_split(dataset, [train_size, test_size])
        return train_dataset, val_dataset
