from abc import ABC, abstractmethod
from typing import Optional, List

from datasets import Dataset
from sentence_transformers import models, SentenceTransformer



class AbstractTask(ABC):

    def __init__(self, config):
        self.config = config
        self.evaluator = None

    @abstractmethod
    def train(self, word_embedding_model: models.Transformer, save_to_dir: str,
            step_metric: Optional[str], report_wandb: bool = False, hp_search_mode: bool = False) -> SentenceTransformer:
        raise NotImplementedError()


    @classmethod
    def create_evaluator(cls, evaluator_config):
        raise NotImplementedError()
