import numpy as np
from sentence_transformers.readers import InputExample
from torch.utils.data import Dataset
from typing import Union


class TripletDataset(Dataset):
    def __init__(self, corpus, disc_ids, n_examples : Union[str, int] = "all", shuffle=True):
        self.corpus = corpus

        duplicate_clusters = dict()
        for i, disc_id in enumerate(disc_ids):
            if disc_id in duplicate_clusters.keys():
                duplicate_clusters[disc_id].append(i)
            else:
                duplicate_clusters[disc_id] = []

        self.triplets = []
        for dupl_series in duplicate_clusters.values():
            if len(dupl_series) <= 1:
                continue
            for i in range(len(dupl_series)):
                anchor = dupl_series[i]
                for j in range(i + 1, len(dupl_series)):
                    pos = dupl_series[j]
                    neg = self.__get_negative_example(duplicate_clusters, len(disc_ids))
                    self.triplets.append((anchor, pos, neg))

        if shuffle:
            np.random.shuffle(self.triplets)

        if n_examples != "all":
            self.triplets = self.triplets[:n_examples]

    def __get_negative_example(self, dupl_series, corpus_size) -> int:
        neg = np.random.randint(corpus_size)
        while neg in dupl_series:
            neg = np.random.randint(corpus_size)
        return neg

    def __getitem__(self, index):
        anchor, pos, neg = self.triplets[index]
        return InputExample(texts=[self.corpus[anchor], self.corpus[pos], self.corpus[neg]])

    def __len__(self):
        return len(self.triplets)
