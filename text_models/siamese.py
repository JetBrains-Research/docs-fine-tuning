import tempfile

import numpy as np
from gensim.test.utils import get_tmpfile
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch import nn
from torch.utils.data import DataLoader

from text_models import AbstractModel
from text_models.bert_tasks import AbstractTask
from text_models.datasets import CosineSimilarityDataset, TripletDataset


class BertSiameseModel(AbstractModel):
    def __init__(
        self,
        corpus=None,
        disc_ids=None,
        finetuning_strategy: AbstractTask = None,
        vector_size=384,
        epochs=5,
        batch_size=16,
        n_examples=None,
        warmup_steps=0.1,
        val_size=0.1,
        task_loss="cossim",  # triplet
        pretrained_model="bert-base-uncased",
        tmp_file=get_tmpfile("pretrained_vectors.txt"),
        device="cpu",
        seed=42,
        save_to_path="./",
        models_suffixes=None,
    ):
        super().__init__(vector_size, epochs, pretrained_model, seed, save_to_path, models_suffixes)
        self.device = device
        self.tmp_file = tmp_file or get_tmpfile("pretrained_vectors.txt")
        self.batch_size = batch_size
        self.warmup_steps = np.ceil(n_examples * self.epochs * warmup_steps)
        self.loss = task_loss
        self.finetuning_strategy = finetuning_strategy

        self.task_dataset = None
        self.evaluator = None
        if corpus is not None and disc_ids is not None:
            sentences = [" ".join(doc) for doc in corpus]
            train_size = int(len(corpus) * (1 - val_size))
            train_corpus = sentences[:train_size]
            train_disc_ids = disc_ids[:train_size]

            val_corpus = sentences[train_size:]
            val_disc_ids = disc_ids[train_size:]

            print(f"Train size = {train_size}, Validation size = {len(val_disc_ids)}")

            self.evaluator = self.__get_evaluator(train_corpus, train_disc_ids, val_corpus, val_disc_ids)
            self.task_dataset = self.__get_dataset(train_corpus, train_disc_ids, n_examples)

        self.vocab_size = None
        self.tokenizer = None

    name = "BERT_SIAMESE"

    def train_from_scratch(self, corpus):
        train_sentences = [" ".join(doc) for doc in corpus]
        dumb_model, tokenizer = self.finetuning_strategy.create_model_from_scratch(train_sentences, self.tmp_file)

        dumb_model_name = tempfile.mkdtemp()
        tokenizer.save_pretrained(dumb_model_name)
        dumb_model.save_pretrained(dumb_model_name)

        word_embedding_model = models.Transformer(dumb_model_name)
        self.__train_siamese(word_embedding_model)

    def train_pretrained(self, corpus):
        word_embedding_model = models.Transformer(self.pretrained_model)
        self.__train_siamese(word_embedding_model)

    def train_finetuned(self, base_corpus, extra_corpus):
        word_embedding_model = self.finetuning_strategy.finetune_on_docs(
            self.pretrained_model, [" ".join(doc) for doc in extra_corpus], self.device, self.save_to_path
        )

        self.__train_siamese(word_embedding_model)

    def __train_siamese(self, word_embedding_model):

        self.model = self._create_sentence_transformer(word_embedding_model)

        train_dataloader = DataLoader(self.task_dataset, shuffle=True, batch_size=self.batch_size)
        train_loss = (
            losses.CosineSimilarityLoss(self.model)
            if self.loss == "cossim"
            else losses.TripletLoss(
                model=self.model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=1
            )
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            evaluator=self.evaluator,
            evaluation_steps=500,
        )

    def __get_dataset(self, corpus, disc_ids, n_examples):
        if self.loss == "cossim":
            return CosineSimilarityDataset(corpus, disc_ids, n_examples, shuffle=True)
        if self.loss == "triplet":
            return TripletDataset(corpus, disc_ids, n_examples, shuffle=True)
        raise ValueError("Unsupported loss")

    def __get_evaluator(self, train_corpus, train_disc_ids, val_corpus, val_disc_ids):
        queries = {qid: query for qid, query in enumerate(val_corpus)}
        corpus = {cid: doc for cid, doc in enumerate(train_corpus)}
        relevant_docs = {
            qid: {cid for cid in corpus.keys() for qid in queries.keys() if train_disc_ids[cid] == val_disc_ids[qid]}
            for qid in queries.keys()
        }

        return InformationRetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            corpus_chunk_size=500,
            batch_size=self.batch_size,
            precision_recall_at_k=[1, 5, 10, 15],
            main_score_function="cos_sim",
        )

    def _create_sentence_transformer(self, word_embedding_model: models.Transformer) -> SentenceTransformer:
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.vector_size,
            activation_function=nn.Tanh(),
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)

    def get_embeddings(self, corpus):
        return self.model.encode([" ".join(report) for report in corpus]).astype(np.float32)

    def get_doc_embedding(self, doc):
        return self.get_embeddings([" ".join(doc)])[0]

    @classmethod
    def load(cls, path):
        sbert_model = SentenceTransformer(path)
        model = BertSiameseModel()
        model.model = sbert_model
        return model

    def save(self, path):
        self.model.save(path)
