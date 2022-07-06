from typing import List, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from data_processing.util import randint_except
from text_models.datasets import BertModelDataset


class NextSentenceDataset(BertModelDataset):
    def __init__(
        self,
        corpus: List[str],
        tokenizer: PreTrainedTokenizerBase,
        n_examples: Union[str, int] = "all",
        max_len: int = 512,
        forget_const: int = 10,
    ):
        sentence_a = []
        sentence_b = []
        label = []
        for i in range(len(corpus) - 1):
            if np.random.rand() >= 0.5:
                sentence_a.append(corpus[i])
                sentence_b.append(corpus[i + 1])
                label.append(0)
            else:
                sentence_a.append(corpus[i])
                excluding = np.concatenate((np.arange(0, i - forget_const), np.arange(i + forget_const, len(corpus))))
                sentence_b.append(corpus[randint_except(0, len(corpus), excluding)])
                label.append(1)

        inputs = tokenizer(
            sentence_a, sentence_b, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length"
        )
        inputs["labels"] = torch.LongTensor([label]).T

        super().__init__(inputs, n_examples)
