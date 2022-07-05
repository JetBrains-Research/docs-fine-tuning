from typing import Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data_processing.util import sections_to_sentences, Corpus
from text_models.bert_tasks.sentences_classification import SentencesClassificationTask
from text_models.datasets import NextSentenceDataset


class NextSentencePredictionTask(SentencesClassificationTask):
    name = "nsp"

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: int = 200,
        n_examples: Union[str, int] = "all",
        save_best_model: bool = False,
        forget_const: int = 10,
        save_steps: int = 2000,
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, save_best_model, save_steps)
        self.forget_const = forget_const

    def _get_dataset(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dataset:
        corpus = sections_to_sentences(corpus)
        return NextSentenceDataset(corpus, tokenizer, self.n_examples, max_len, self.forget_const)
