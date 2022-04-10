import numpy as np
import pandas as pd
import faiss

from approaches import AbstractApproach


class SimpleApproach(AbstractApproach):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, test)

    def additional_preparation(self):
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def get_dupl_ids(self, query_num, topn: int):
        return self.index.search(self.test_embs[query_num].reshape(1, -1), topn)[1][0]

    def update_history(self, query_num):
        tmp_emb = np.array([self.test_embs[query_num]])
        faiss.normalize_L2(tmp_emb)
        self.index.add(tmp_emb)
