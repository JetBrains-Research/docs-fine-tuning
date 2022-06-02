import os
from typing import List

import numpy as np
from sentence_transformers import models, InputExample, losses, SentenceTransformer, evaluation
from torch.utils.data import DataLoader

from data_processing.util import sections_to_sentences
from text_models.bert_tasks import AbstractTask


class STSTask(AbstractTask):
    name = "sts"

    def __init__(
        self,
        epochs=2,
        batch_size=16,
        eval_steps=200,
        n_examples="all",
        save_best_model=False,
        warmup_steps=0.1,
        forget_const=10,
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, save_best_model)
        self.forget_const = forget_const
        self.warmup_steps = warmup_steps

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: List[List[List[str]]],
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ):
        corpus = sections_to_sentences(docs_corpus)

        word_embedding_model = models.Transformer(pretrained_model)
        train_dataloader = self.__get_train_dataloader_from_docs(corpus)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        train_loss = losses.CosineSimilarityLoss(model)

        warmup = len(train_dataloader) * self.epochs * self.warmup_steps
        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")
        output_path = os.path.join(save_to_path, "output_docs")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup,
            evaluator=evaluator,
            evaluation_steps=self.eval_steps,
            checkpoint_path=checkpoints_path,
            output_path=output_path,
            checkpoint_save_total_limit=3,
            save_best_model=self.save_best_model,
        )

        if self.save_best_model:
            model = SentenceTransformer(output_path)

        return model._first_module()

    def __get_train_dataloader_from_docs(self, docs_corpus):
        train_data = []
        corpus = [" ".join(doc) for doc in docs_corpus]
        lngth = len(docs_corpus) - 1
        for i in range(lngth):
            train_data.append(InputExample(texts=[corpus[i], corpus[i + 1]], label=1.0))
            if i + self.forget_const < lngth:
                train_data.append(
                    InputExample(
                        texts=[corpus[i], corpus[i + np.random.randint(self.forget_const, lngth - i)]], label=0.0
                    )
                )

        return DataLoader(train_data[: self.n_examples], shuffle=True, batch_size=self.batch_size)
