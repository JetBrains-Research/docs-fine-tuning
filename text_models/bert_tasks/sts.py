import os
from typing import List, Union, Optional

import numpy as np
from sentence_transformers import models, InputExample, losses, SentenceTransformer, evaluation
from torch.utils.data import DataLoader, Dataset

from data_processing.util import sections_to_sentences
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.evaluation import LossEvaluator, ValMetric, WandbLoggingEvaluator


class STSTask(AbstractTask):
    """
    Semantic Textual Similarity Task. We assume that two adjacent sentences are similar in meaning.

    :param epochs: Number of fine-tuning epochs
    :param batch_size: Batch size used for fine-tuning
    :param eval_steps: Number of update steps between two evaluations
    :param n_examples: Number of input examples that will be used for fine-tuning
    :param save_best_model: Whether or not to save the best model found during training at the end of training
    :param forget_const: Negative example is chosen as a random sentence in range [(i + forget_const)..len(corpus))
    """

    name = "sts"

    class ListDataset(Dataset):
        def __init__(self, data_list: list):
            self.data = data_list

        def __getitem__(self, item):
            return self.data[item]

        def __len__(self):
            return len(self.data)

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: Optional[int] = None,
        n_examples: Union[str, int] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_best_model: bool = False,
        save_steps: Optional[int] = None,
        do_eval_on_artefacts: bool = True,
        max_len: Optional[int] = None,
        warmup_ratio: float = 0.,
        weight_decay: float = 0.,
        forget_const: int = 10,
    ):
        super().__init__(
            epochs,
            batch_size,
            eval_steps,
            n_examples,
            val,
            metric_for_best_model,
            save_steps,
            save_best_model,
            do_eval_on_artefacts,
            max_len,
            warmup_ratio,
            weight_decay
        )
        self.forget_const = forget_const

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: List[List[List[str]]],
        evaluator: evaluation.InformationRetrievalEvaluator,
        device: str,
        save_to_path: str,
        report_wandb: bool = False,
    ) -> models.Transformer:
        corpus = sections_to_sentences(docs_corpus)

        word_embedding_model = models.Transformer(pretrained_model, max_seq_length=self.max_len)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

        dataset = self.__get_train_data_from_docs(corpus)

        train_dataset, val_dataset = self._train_val_split(dataset)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size)  # type: ignore

        train_loss = losses.CosineSimilarityLoss(model)
        evaluator = LossEvaluator(
            evaluator,
            train_loss,
            val_dataset,
            STSTask.ListDataset(evaluator.val_dataset),  # type: ignore
            self.metric_for_best_model,
            self.batch_size,
        )
        if report_wandb:
            evaluator = WandbLoggingEvaluator(evaluator, f"{self.name}/global_steps", len(train_dataloader))

        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")
        output_path = os.path.join(save_to_path, "output_docs")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=np.ceil(len(train_dataloader) * self.epochs * self.warmup_ratio),
            weight_decay=self.weight_decay,
            evaluator=evaluator,
            evaluation_steps=0 if self.eval_steps is None else self.eval_steps,
            checkpoint_path=checkpoints_path,
            output_path=output_path,
            checkpoint_save_steps=self.save_steps,
            save_best_model=self.save_best_model,
        )

        if self.save_best_model:
            model = SentenceTransformer(output_path)

        return model._first_module()

    def __get_train_data_from_docs(self, docs_corpus: List[str]) -> ListDataset:
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
        n_examples = len(train_data) if self.n_examples == "all" else int(self.n_examples)

        return STSTask.ListDataset(train_data[:n_examples])

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        model = SentenceTransformer(load_from_path)
        return model._first_module()
