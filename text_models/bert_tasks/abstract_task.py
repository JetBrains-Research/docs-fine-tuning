from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import models, evaluation

from data_processing.util import flatten


class AbstractTask(ABC):
    def __init__(
        self, epochs=2, batch_size=16, eval_steps=200, n_examples="all", save_best_model=False, name="abstract"
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.n_examples = n_examples
        self.eval_steps = eval_steps
        self.save_best_model = save_best_model

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

    @staticmethod
    def sections_to_sentences(docs_corpus):
        return [" ".join(doc) for doc in flatten(docs_corpus)]
