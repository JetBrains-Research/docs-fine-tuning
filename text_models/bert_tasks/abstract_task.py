from abc import ABC, abstractmethod
from typing import List, Union

from sentence_transformers import models, evaluation


class AbstractTask(ABC):
    name = "abstract"

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: int = 200,
        n_examples: Union[str, int] = "all",
        save_best_model: bool = False,
        load_from_path: Union[None, str] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.eval_steps = eval_steps
        self.save_best_model = save_best_model
        self.load_from_path = load_from_path

    @abstractmethod
    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: List[List[List[str]]],  # list of list(sections) of list(sentences) of tokens(words)
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:
        raise NotImplementedError()

    @abstractmethod
    def load(self) -> models.Transformer:
        raise NotImplementedError()
