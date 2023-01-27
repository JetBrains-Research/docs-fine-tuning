import logging
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from text_models.bert_tasks.evaluation.save_loss import write_csv_loss
from text_models.bert_tasks.evaluation import ValMetric

logger = logging.getLogger(__name__)


class LossEvaluator(SentenceEvaluator):
    def __init__(
        self,
        base_evaluator: SentenceEvaluator,
        loss: Module,
        eval_dataset: Dataset,
        eval_task_dataset: Dataset,
        metric_for_best_model: str = ValMetric.TASK,
        batch_size: int = 32,
    ):
        self.base_evaluator = base_evaluator
        self.loss = loss
        self.eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)
        self.eval_task_dataloader = DataLoader(eval_task_dataset, shuffle=True, batch_size=batch_size)
        self.metric_for_best_model = metric_for_best_model

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        self.eval_dataloader.collate_fn = model.smart_batching_collate
        self.eval_task_dataloader.collate_fn = model.smart_batching_collate

        loss_value = self.__compute_loss(self.eval_dataloader)
        loss_task_value = self.__compute_loss(self.eval_task_dataloader)

        logger.info(f"Epoch: {epoch} Step: {steps} Eval Loss: {loss_value} Eval Loss on Task Data: {loss_task_value}")
        if output_path is not None:
            write_csv_loss(loss_value, loss_task_value, output_path, epoch, steps)

        base_metric = self.base_evaluator(model, output_path, epoch, steps)
        logger.info(f"Epoch: {epoch} Step: {steps} Base Metric: {base_metric}")

        if self.metric_for_best_model == ValMetric.TASK:
            return base_metric
        if self.metric_for_best_model == ValMetric.LOSS_DOCS:
            return -loss_value  # return minus loss because of specific implementation of SBERT evaluation
        return -loss_task_value

    def __compute_loss(self, dataloader: DataLoader):
        val_loss = 0.0
        with torch.no_grad():
            for data in dataloader:
                features, labels = data
                val_loss += self.loss(features, labels)

        val_loss /= len(dataloader)
        return val_loss.item()  # type: ignore
