from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import models, evaluation


class AbstractTask(ABC):
    def __init__(self, epochs=2, batch_size=16, eval_steps=200, n_examples="all", name="abstract"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.n_examples = n_examples
        self.eval_steps = eval_steps

    @abstractmethod
    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: List[str],
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:
        raise NotImplementedError()
