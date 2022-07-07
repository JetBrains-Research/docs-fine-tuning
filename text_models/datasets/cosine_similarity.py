from typing import Union, List, Tuple

import numpy as np
from sentence_transformers.readers import InputExample
from torch.utils.data import Dataset


class CosineSimilarityDataset(Dataset):
    def __init__(
        self, corpus: List[str], disc_ids: List[str], n_examples: Union[str, int] = "all", shuffle: bool = False
    ):
        self.disc_ids = disc_ids
        self.corpus = corpus

        self.total_examples = self.__collect_examples()
        if shuffle:
            np.random.shuffle(self.total_examples)

        self.n_examples = n_examples
        if n_examples == "all" or len(self.total_examples) < int(n_examples):
            self.n_examples = len(self.total_examples)
        else:
            self.total_examples = self.total_examples[: int(n_examples)]

    def __getitem__(self, index):
        i, j = self.total_examples[index]
        label = 1.0 if self.disc_ids[i] == self.disc_ids[j] else 0.0
        return InputExample(texts=[self.corpus[i], self.corpus[j]], label=label)

    def __len__(self):
        return self.n_examples

    def __collect_examples(self) -> List[Tuple[int, int]]:
        pos_examples = [
            (i, j)
            for i in range(len(self.corpus))
            for j in range(i + 1, len(self.corpus))
            if self.disc_ids[i] == self.disc_ids[j]
        ]
        np.random.shuffle(pos_examples)

        def neg_generator():
            seen = set()
            x, y = 0, 0

            while True:
                while self.disc_ids[x] == self.disc_ids[y] or (x, y) in seen:
                    x, y = np.random.randint(len(self.disc_ids), size=2)
                yield (x, y)
                seen.add((x, y))

        neg_gen = neg_generator()
        neg_examples = [next(neg_gen) for _ in range(len(pos_examples))]

        return pos_examples + neg_examples
