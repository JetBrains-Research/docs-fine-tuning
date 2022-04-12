import tempfile
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset

from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample

from gensim.test.utils import get_tmpfile

from text_models.abstract_model import AbstractModel
from text_models.bert import BertModelMLM


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
        i = 0
        cur_bound = 0
        while index > cur_bound:
            i += 1
            cur_bound += i + 1
        j = index - i * (i + 1) // 2
        i += 1

        label = 1.0 if self.disc_ids[i] == self.disc_ids[j] else 0.0
        return InputExample(texts=[self.corpus[i], self.corpus[j]], label=label)

    def __len__(self):
        return self.n_examples


class SBertModel(AbstractModel):
    def __init__(
        self,
        corpus=None,
        disc_ids=None,
        vector_size=256,
        epochs=2,
        batch_size=16,
        warmup_steps=0.1,
        max_len=512,
        forget_const=10,
        tmp_file=get_tmpfile("pretrained_vectors.txt"),
        n_examples=None,
        pretrained_model="all-mpnet-base-v2",
        seed=42,
        save_to_path="./",
        models_suffixes=None,
    ):
        super().__init__(vector_size, epochs, pretrained_model, seed, save_to_path, models_suffixes)
        self.tmp_file = tmp_file or get_tmpfile("pretrained_vectors.txt")
        self.batch_size = batch_size

        if corpus is not None and disc_ids is not None:
            self.train_sts_dataloader = self.__get_train_dataloader_from_reports(corpus, disc_ids, n_examples)
            self.warmup_steps = int(len(self.train_sts_dataloader) * self.epochs * warmup_steps)

        self.max_len = max_len
        self.forget_const = forget_const

    name = "sbert"

    def train_from_scratch(self, corpus):
        train_sentences = [" ".join(doc) for doc in corpus]
        dumb_model, tokenizer = BertModelMLM.create_bert_model(train_sentences, self.tmp_file, self.max_len, task="sts")

        dumb_model_name = tempfile.mkdtemp()
        tokenizer.save_pretrained(dumb_model_name)
        dumb_model.save_pretrained(dumb_model_name)

        word_embedding_model = models.Transformer(dumb_model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.vector_size,
            activation_function=nn.Tanh(),
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        self.__train_sts(self.train_sts_dataloader)

    def train_pretrained(self, corpus):
        self.model = SentenceTransformer(self.pretrained_model)
        self.__train_sts(self.train_sts_dataloader)

    def train_finetuned(self, base_corpus, extra_corpus):
        self.model = SentenceTransformer(self.pretrained_model)
        extra_train_dataloader = self.__get_train_dataloader_from_docs(extra_corpus)
        self.__train_sts(extra_train_dataloader)
        self.__train_sts(self.train_sts_dataloader)

    def __train_sts(self, train_dataloader):
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps
        )

    def get_embeddings(self, corpus):
        return self.model.encode([" ".join(report) for report in corpus]).astype(np.float32)

    def get_doc_embedding(self, doc):
        return self.get_embeddings([" ".join(doc)])[0]

    @classmethod
    def load(cls, path):
        sbert_model = SentenceTransformer(path)
        model = SBertModel()
        model.model = sbert_model
        return model

    def save(self, path):
        self.model.save(path)

    def __get_train_dataloader_from_reports(self, corpus, disc_ids, n_examples):
        corpus = list(map(lambda x: " ".join(x), corpus))
        train_data = SbertModelDataset(corpus, disc_ids, n_examples, shuffle=True)
        return DataLoader(train_data, shuffle=True, batch_size=self.batch_size)

    def __get_train_dataloader_from_docs(self, docs_corpus):
        train_data = []
        corpus = list(map(lambda x: " ".join(x), docs_corpus))
        lngth = len(docs_corpus) - 1
        for i in range(lngth):
            train_data.append(InputExample(texts=[corpus[i], corpus[i + 1]], label=1.0))
            if i + self.forget_const < lngth:
                train_data.append(
                    InputExample(
                        texts=[corpus[i], corpus[i + np.random.randint(self.forget_const, lngth - i)]], label=0.0
                    )
                )

        return DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
