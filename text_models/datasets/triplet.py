from typing import Union, List, Dict

import numpy as np
from sentence_transformers.readers import InputExample
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """
    Dataset for Siamese Neural Network with Triplet Loss.

    :param corpus: Corpus of bug report descriptions
    :param disc_ids: List where position i is the id of the oldest bug report that is a duplicate of i-th bug report
    :param n_examples: Number of examples used in dataset
    :param suffle: Shuffle or not the resulting dataset
    """

    def __init__(
        self, corpus: List[str], disc_ids: List[str], n_examples: Union[str, int] = "all", shuffle: bool = True
    ):
        self.corpus = corpus

        duplicate_clusters: Dict[str, List[int]] = dict()
        for i, disc_id in enumerate(disc_ids):
            if disc_id in duplicate_clusters.keys():
                duplicate_clusters[disc_id].append(i)
            else:
                duplicate_clusters[disc_id] = [i]

        self.triplets = []
        for dupl_series in duplicate_clusters.values():
            n_dupl = len(dupl_series)
            if n_dupl <= 1:
                continue

            old_negs = dict()
            for i in range(n_dupl):
                anchor = dupl_series[i]
                for j in range(n_dupl):
                    if j == i:
                        continue
                    pos = dupl_series[j]
                    neg = self.__get_negative_example(dupl_series, len(disc_ids), old_negs.get(pos))
                    self.triplets.append((anchor, pos, neg))
                    old_negs[anchor] = neg

        if shuffle:
            np.random.shuffle(self.triplets)

        if n_examples != "all":
            self.triplets = self.triplets[: int(n_examples)]

    def __get_negative_example(self, dupl_series, corpus_size, old_neg) -> int:
        neg = np.random.randint(corpus_size)
        while neg in dupl_series or neg == old_neg:
            neg = np.random.randint(corpus_size)
        return neg

    def __getitem__(self, index):
        anchor, pos, neg = self.triplets[index]
        return InputExample(texts=[self.corpus[anchor], self.corpus[pos], self.corpus[neg]])

    def __len__(self):
        return len(self.triplets)