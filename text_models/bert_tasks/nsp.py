from typing import Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data_processing.util import sections_to_sentences, Corpus
from text_models.bert_tasks.sentences_classification import SentencesClassificationTask
from text_models.datasets import NextSentenceDataset


class NextSentencePredictionTask(SentencesClassificationTask):
    """
    Next Sentence Prediction task.

    :param epochs: Number of fine-tuning epochs
    :param batch_size: Batch size used for fine-tuning
    :param eval_steps: Number of update steps between two evaluations
    :param n_examples: Number of input examples that will be used for fine-tuning
    :param save_best_model: Whether or not to save the best model found during training at the end of training
    :param forget_const: Negative example is chosen as a random sentence in range 0..len(corpus) excluding [i - forget_const, i + forget_const]
    :param save_steps: Number of updates steps before two checkpoint saves
    """

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
        sentences = sections_to_sentences(corpus)
        return NextSentenceDataset(sentences, tokenizer, self.n_examples, max_len, self.forget_const)
