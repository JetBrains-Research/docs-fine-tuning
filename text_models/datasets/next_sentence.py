from typing import List
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class NextSentenceDataset(Dataset):
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

        self.inputs = tokenizer(
            sentence_a, sentence_b, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length"
        )
        self.inputs["labels"] = torch.LongTensor([label]).T

        self.n_examples = n_examples
        if n_examples == "all" or n_examples > len(self.inputs.input_ids):
            self.n_examples = len(self.inputs.input_ids)

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.inputs.items()}

    def __len__(self):
        return self.n_examples
