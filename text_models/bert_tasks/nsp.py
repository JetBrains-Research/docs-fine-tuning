from typing import Union, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data_processing.util import sections_to_sentences, Corpus
from text_models.bert_tasks.evaluation import ValMetric
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
        eval_steps: Optional[int] = None,
        n_examples: Union[str, int] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_best_model: bool = False,
        forget_const: int = 10,
        save_steps: Optional[int] = None,
        do_eval_on_artefacts: bool = True,
        max_len: Optional[int] = None,
    ):
        super().__init__(
            epochs,
            batch_size,
            eval_steps,
            n_examples,
            val,
            metric_for_best_model,
            save_steps,
            save_best_model,
            do_eval_on_artefacts,
            max_len
        )
        self.forget_const = forget_const

    def _get_dataset(self, corpus: Corpus, tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dataset:
        sentences = sections_to_sentences(corpus)
        return NextSentenceDataset(sentences, tokenizer, self.n_examples, max_len, self.forget_const)
