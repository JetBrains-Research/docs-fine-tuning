import logging
from typing import Optional, Dict

import torch
from sentence_transformers.evaluation import SentenceEvaluator
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from text_models.evaluation.util import write_csv_loss, ValMetric
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LossEvaluator(SentenceEvaluator):
    def __init__(
        self,
        base_evaluator: SentenceEvaluator,
        loss: Module,
        eval_dataset: Optional[Dataset],
        eval_task_dataset: Dataset,
        metric_for_best_model: str = ValMetric.TASK,
        batch_size: int = 32,
    ):
        super(LossEvaluator, self).__init__()
        self.base_evaluator = base_evaluator
        self.loss = loss
        self.eval_dataloader = (
            DataLoader(eval_dataset, shuffle=True, batch_size=batch_size) if eval_dataset is not None else None
        )
        self.eval_task_dataloader = DataLoader(eval_task_dataset, shuffle=True, batch_size=batch_size)
        self.metric_for_best_model = metric_for_best_model

        self.metrics: Optional[Dict[str, float]] = None

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        loss_value = -1
        if self.eval_dataloader is not None:
            self.eval_dataloader.collate_fn = model.smart_batching_collate
            loss_value = self.__compute_loss(self.eval_dataloader)

        self.eval_task_dataloader.collate_fn = model.smart_batching_collate
        loss_task_value = self.__compute_loss(self.eval_task_dataloader)

        if loss_value != -1:
            logger.info(
                f"Epoch: {epoch} Step: {steps} Eval Loss: {loss_value} Eval Loss on Task Data: {loss_task_value}"
            )
        if output_path is not None:
            write_csv_loss(loss_value, loss_task_value, output_path, epoch, steps)
        base_metric = self.base_evaluator(model, output_path, epoch, steps)
        logger.info(f"Epoch: {epoch} Step: {steps} Base Metric: {base_metric}")

        self.metrics = {
            ValMetric.TASK: base_metric,
            ValMetric.LOSS_TASK: loss_task_value,
            ValMetric.LOSS_DOCS: loss_value,
        }

        if self.metric_for_best_model == ValMetric.TASK:
            return base_metric
        if self.metric_for_best_model == ValMetric.LOSS_DOCS:
            return -loss_value  # return minus loss because of specific implementation of SBERT evaluation
        return -loss_task_value

    def __compute_loss(self, dataloader: DataLoader):
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(dataloader):
                features, labels = data
                val_loss += self.loss(features, labels)

        val_loss /= len(dataloader)
        return val_loss.item()  # type: ignore
