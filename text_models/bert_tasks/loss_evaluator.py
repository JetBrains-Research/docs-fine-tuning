import logging
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class LossEvaluator(SentenceEvaluator):
    def __init__(self, base_evaluator: SentenceEvaluator, loss: Module, eval_dataset: Dataset, batch_size: int = 32):
        self.base_evaluator = base_evaluator
        self.loss = loss
        self.eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)

    def __call__(self,  model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        base_metric = self.base_evaluator(model, output_path, epoch, steps)
        logger.info(f"Epoch: {epoch} Step: {steps} Base Metric: {base_metric}")

        self.eval_dataloader.collate_fn = model.smart_batching_collate

        val_loss = 0
        with torch.no_grad():
            for data in self.eval_dataloader:
                features, labels = data
                val_loss += self.loss(features, labels)

        val_loss /= len(self.eval_dataloader)

        logger.info(f"Epoch: {epoch} Step: {steps} Eval Loss: {val_loss.item()}")

        return -val_loss.item() # return minus average loss because of specific implementation of SBERT evaluation




