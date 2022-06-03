from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data_processing.util import Sections
from text_models.bert_tasks.sentences_classification import SentencesClassificationTask
from text_models.datasets import SameSectionDataset


class SameSectionTask(SentencesClassificationTask):
    name = "sase"

    def _get_dataset(self, corpus: Sections, tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dataset:
        sections = [[" ".join(sentence) for sentence in section] for section in corpus]
        return SameSectionDataset(sections, tokenizer, self.n_examples, max_len)
