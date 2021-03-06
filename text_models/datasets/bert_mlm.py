from typing import Union

import numpy as np
import torch

from text_models.datasets import BertModelDataset


class BertModelMLMDataset(BertModelDataset):
    """
    Dataset for BERT model computations on Masked Language Modeling (MLM) task.

    Inherits from BertModelDataset.

    :param encodings: BERT tokenizer results
    :param mask_id: <MASK> token ID
    :param cls_id: <CLS> token ID
    :param sep_id: <SEP> token ID
    :param pad_id: <PAD> token ID
    :param mask_probability: Probability that some word will be replaced by a <MASK> token
    :param n_examples: Number of examples used in dataset
    """

    def __init__(
        self,
        encodings,
        mask_id: int = 103,
        cls_id: int = 102,
        sep_id: int = 101,
        pad_id: int = 0,
        mask_probability: float = 0.15,
        n_examples: Union[str, int] = "all",
    ):
        super().__init__(encodings, n_examples)

        self.encodings["labels"] = self.encodings.input_ids.detach().clone()
        self.mask_proba = mask_probability
        self.mask_id = mask_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.masked = np.full(self.__len__(), False)

    def __getitem__(self, idx):
        if not self.masked[idx]:
            inputs = self.encodings.input_ids[idx]

            rand = torch.rand(inputs.shape)
            mask_arr = (
                (rand < self.mask_proba) * (inputs != self.cls_id) * (inputs != self.sep_id) * (inputs != self.pad_id)
            )
            inputs[mask_arr] = self.mask_id

            self.encodings.input_ids[idx] = inputs
            self.masked[idx] = True

        return super().__getitem__(idx)
