from typing import List

import numpy as np
from sentence_transformers import models, InputExample, losses, SentenceTransformer
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

from data_processing.util import get_corpus_properties
from text_models.bert_tasks import AbstractTask


class STSTask(AbstractTask):
    def __init__(self, epochs=2, batch_size=16, n_examples="all", max_len=512, warmup_steps=0.1, forget_const=10):
        super().__init__(epochs, batch_size, n_examples, max_len, name="sts")
        self.forget_const = forget_const
        self.warmup_steps = warmup_steps

    def create_model_from_scratch(self, train_sentences: List[str], tmp_file: str):
        vocab_size, _ = get_corpus_properties([sentence.split(" ") for sentence in train_sentences])

        tokenizer = self._create_and_train_tokenizer(train_sentences, vocab_size, tmp_file)

        bert_config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=self.max_len + 2,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        dumb_model = BertModel(bert_config)

        return dumb_model, tokenizer

    def finetune_on_docs(self, pretrained_model: str, docs_corpus: List[str], device: str, save_to_path: str):
        word_embedding_model = models.Transformer(pretrained_model)
        train_dataloader = self.__get_train_dataloader_from_docs(docs_corpus)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        train_loss = losses.CosineSimilarityLoss(model)

        warmup = len(train_dataloader) * self.epochs * self.warmup_steps
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=warmup)
        return model._first_module()

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
