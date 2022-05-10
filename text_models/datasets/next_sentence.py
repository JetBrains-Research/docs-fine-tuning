from typing import List
from typing import Union

import numpy as np
import torch

from text_models.datasets import BertModelDataset


class NextSentenceDataset(BertModelDataset):
    def __init__(self, corpus: List[str], tokenizer, n_examples: Union[str, int] = "all", max_len=512, forget_const=10):
        sentence_a = []
        sentence_b = []
        label = []
        lngth = len(corpus) - 1
        for i in range(lngth - forget_const):
            if np.random.rand() >= 0.5:
                sentence_a.append(corpus[i])
                sentence_b.append(corpus[i + 1])
                label.append(1)
            else:
                sentence_a.append(corpus[i])
                sentence_b.append(corpus[i + np.random.randint(forget_const, lngth - i)])
                label.append(0)

        inputs = tokenizer(
            sentence_a, sentence_b, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length"
        )
        inputs["labels"] = torch.LongTensor([label]).T

        super().__init__(inputs, n_examples)
