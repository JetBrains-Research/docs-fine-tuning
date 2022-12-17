import logging
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from text_models.bert_tasks.evaluation.save_loss import write_csv_loss

logger = logging.getLogger(__name__)

class LossEvaluator(SentenceEvaluator):
    def __init__(self, base_evaluator: SentenceEvaluator, loss: Module, eval_dataset: Dataset, batch_size: int = 32):
        self.base_evaluator = base_evaluator
        self.loss = loss
        self.eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)

    def __call__(self,  model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        self.eval_dataloader.collate_fn = model.smart_batching_collate

        val_loss = 0.
        with torch.no_grad():
            for data in self.eval_dataloader:
                features, labels = data
                val_loss += self.loss(features, labels)

        val_loss /= len(self.eval_dataloader)

        loss_value = val_loss.item() # type: ignore

        logger.info(f"Epoch: {epoch} Step: {steps} Eval Loss: {loss_value}")
        if output_path is not None:
            write_csv_loss(loss_value, output_path, epoch, steps)

        base_metric = self.base_evaluator(model, output_path, epoch, steps)
        logger.info(f"Epoch: {epoch} Step: {steps} Base Metric: {base_metric}")

        return -loss_value # return minus average loss because of specific implementation of SBERT evaluation
