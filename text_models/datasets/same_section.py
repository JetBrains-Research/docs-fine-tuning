from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from data_processing.util import randint_except
from text_models.datasets import BertModelDataset


class SameSectionDataset(BertModelDataset):
    """
    Dataset for BERT model computations on Same Section (SaSe) task.

    SaSe is a slight modification of the Next Sentence Prediction task, where negative examples are taken only from different sections of the documentation.

    :param corpus: Corpus of documentation sections
    :param tokenizer: Tokenizer that will be used to tokenize bug report descriptions
    :param n_examples: Number of examples used in dataset
    :param max_len: max_length parameter of tokenizer
    """

    def __init__(
        self,
        sections: List[List[str]],
        tokenizer: PreTrainedTokenizerBase,
        n_examples: Union[str, int] = "all",
        max_len: int = 512,
    ):
        sentences_a = []
        sentences_b = []

        labels = []
        for i, section in enumerate(sections):
            section_lngth = len(section)
            for j in range(section_lngth - 1):
                if np.random.rand() >= 0.5:
                    sentences_a.append(section[j])
                    sentences_b.append(section[j + 1])
                    labels.append(0)
                else:
                    sentences_a.append(section[j])
                    random_section = sections[randint_except(0, len(sections), [i])]
                    neg_example = random_section[np.random.randint(0, len(random_section))]
                    sentences_b.append(neg_example)
                    labels.append(1)

        inputs = tokenizer(
            sentences_a, sentences_b, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length"
        )
        inputs["labels"] = torch.LongTensor([labels]).T
        super().__init__(inputs, n_examples)
