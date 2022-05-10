from typing import Union

from torch.utils.data import Dataset


class BertModelDataset(Dataset):
    def __init__(self, encodings, n_examples: Union[str, int] = "all"):
        self.encodings = encodings

        self.n_examples = n_examples
        if n_examples == "all" or n_examples > len(self.encodings.input_ids):
            self.n_examples = len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return self.n_examples
