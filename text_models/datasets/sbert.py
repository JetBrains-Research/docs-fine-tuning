import numpy as np

from torch.utils.data import Dataset
from sentence_transformers.readers import InputExample

class SbertModelDataset(Dataset):
    def __init__(self, corpus, disc_ids, n_examples, shuffle=False):
        if shuffle:
            data = list(zip(corpus, disc_ids))
            np.random.shuffle(data)
            corpus, disc_ids = list(zip(*data))

        self.disc_ids = list(disc_ids)
        self.corpus = list(corpus)
        self.n_examples = n_examples

    def __getitem__(self, index):

        def index_to_pair():
            """
            Convert index in array of distinct bug_report pairs into a pair of indexes in bug_report triangle matrix
            :return: a pair of indexes in bug_report triangle matrix
            """
            i = 0
            cur_bound = 0
            while index > cur_bound:
                i += 1
                cur_bound += i + 1
            j = index - i * (i + 1) // 2
            return i + 1, j

        i, j = index_to_pair()

        label = 1.0 if self.disc_ids[i] == self.disc_ids[j] else 0.0
        return InputExample(texts=[self.corpus[i], self.corpus[j]], label=label)

    def __len__(self):
        return self.n_examples