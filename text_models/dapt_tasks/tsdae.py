import os.path
from typing import Union, Optional

import numpy as np
from sentence_transformers import models, losses, SentenceTransformer, evaluation
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader

from data_processing.util import sections_to_sentences, Corpus
from text_models.dapt_tasks import AbstractPreTrainingTask
from text_models.dapt_tasks.evaluation import LossEvaluator, WandbLoggingEvaluator, ValMetric


class TSDenoisingAutoEncoderTask(AbstractPreTrainingTask):
    """
    Transformer-based Denoising AutoEncoder.
    """

    name = "tsdae"

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: Optional[int] = None,
        n_examples: Union[str, int] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_steps: Optional[int] = None,
        save_best_model: bool = False,
        do_eval_on_artefacts: bool = True,
        max_len: int = 512,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.0,
        pooling_mode: str = "mean",
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
            weight_decay,
        )
        self.pooling_mode = pooling_mode

    def train_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,
        evaluator: evaluation.InformationRetrievalEvaluator,
        device: str,
        save_to_path: str,
        report_wandb: bool = False,
    ) -> models.Transformer:
        corpus = sections_to_sentences(docs_corpus)

        word_embedding_model = models.Transformer(pretrained_model, max_seq_length=self.max_len)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode=self.pooling_mode
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

        if self.n_examples == "all":
            self.n_examples = len(corpus)

        dataset = DenoisingAutoEncoderDataset(corpus[: int(self.n_examples)])
        val_task_dataset = DenoisingAutoEncoderDataset(evaluator.val_dataset)  # type: ignore

        train_dataset, val_dataset = self._train_val_split(dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)  # type: ignore

        train_loss = losses.DenoisingAutoEncoderLoss(
            model, decoder_name_or_path=pretrained_model, tie_encoder_decoder=True
        )

        evaluator = LossEvaluator(
            evaluator, train_loss, val_dataset, val_task_dataset, self.metric_for_best_model, self.batch_size
        )

        if report_wandb:
            evaluator = WandbLoggingEvaluator(evaluator, f"{self.name}/global_steps", len(train_dataloader))

        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")
        output_path = os.path.join(save_to_path, "output_docs")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_steps=np.ceil(len(train_dataloader) * self.epochs * self.warmup_ratio),
            scheduler="constantlr",
            optimizer_params={"lr": 3e-5},
            show_progress_bar=True,
            evaluator=evaluator,
            evaluation_steps=0 if self.eval_steps is None else self.eval_steps,
            checkpoint_path=checkpoints_path,
            output_path=output_path,
            checkpoint_save_steps=self.save_steps if self.save_steps is not None else len(train_dataloader),
            save_best_model=self.save_best_model,
        )

        if self.save_best_model:
            model = SentenceTransformer(output_path)

        return model._first_module()

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        model = SentenceTransformer(load_from_path)
        return model._first_module()
