import csv
import logging
import os
from typing import List, Optional, Dict

from tqdm import tqdm

import numpy as np
import torch
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import (
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
    multiclass_accuracy,
)

logger = logging.getLogger(__name__)


class AssigneeMetrics:
    ACC = "Acc@k"  # == w_recall@k
    WAF1 = "w_f1@k"
    W_PRECISION = "w_precision@k"
    RECALL = "recall@k"
    PRECISION = "precision@k"
    F1 = "f1@k"


class AssignmentEvaluator(SentenceEvaluator):
    def __init__(
        self,
        dataset: Dataset,
        num_labels: int,
        val_corpus: Optional[List[str]] = None,
        acc_at_k: List[int] = [1, 5, 10],
        f1_at_k: List[int] = [1, 5, 10],
        recall_at_k: List[int] = [1, 5, 10],
        precision_at_k: List[int] = [1, 5, 10],
        batch_size=16,
        write_csv: bool = True,
    ):
        self.dataset = dataset
        self.val_dataset = val_corpus
        self.batch_size = batch_size
        self.num_labels = num_labels

        self.acc_at_k = acc_at_k
        self.f1_at_k = f1_at_k
        self.recall_at_k = recall_at_k
        self.precision_at_k = precision_at_k

        self.write_csv = write_csv
        self.csv_file = "assignment_evaluation_results.csv"
        self.csv_headers = (
            ["epoch", "steps"]
            + [f"acc_at_{k}" for k in acc_at_k]
            + [f"precision_at_{k}" for k in precision_at_k]
            + [f"w_precision_at_{k}" for k in precision_at_k]
            + [f"recall_at_{k}" for k in recall_at_k]
            + [f"f1_at_{k}" for k in f1_at_k]
            + [f"w_f1_at_{k}" for k in f1_at_k]
        )

        self.softmax_model: Optional[nn.Module] = None

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        metrics = self.compute_metrics(model)

        f1 = metrics[AssigneeMetrics.F1]
        w_f1 = metrics[AssigneeMetrics.WAF1]
        precision = metrics[AssigneeMetrics.PRECISION]
        w_precision = metrics[AssigneeMetrics.W_PRECISION]
        recall = metrics[AssigneeMetrics.RECALL]
        accuracy = metrics[AssigneeMetrics.ACC]

        for name, value in metrics.items():
            logger.info(f"{name}: {value}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps] + accuracy + precision + w_precision + recall + f1 + w_f1)  # type: ignore
            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps] + accuracy + precision + w_precision + recall + f1 + w_f1)  # type: ignore

        return w_f1[np.argmin(self.f1_at_k)]

    def compute_metrics(self, model) -> Dict[str, List[float]]:
        model.eval()

        preds_list = []
        labels_list = []

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=model.smart_batching_collate)
        for batch in tqdm(dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                prediction = model(features[0])["sentence_embedding"]  # type: ignore
            preds_list.append(prediction.softmax(dim=1).to(model.device))
            labels_list.append(label_ids)

        preds = torch.cat(preds_list).to(model.device)
        labels = torch.cat(labels_list).to(model.device)

        metrics = {
            AssigneeMetrics.ACC: [
                multiclass_accuracy(preds, labels, self.num_labels, "weighted", k).item() for k in self.acc_at_k
            ],
            AssigneeMetrics.PRECISION: [
                multiclass_precision(preds, labels, self.num_labels, top_k=k).item() for k in self.precision_at_k
            ],
            AssigneeMetrics.W_PRECISION: [
                multiclass_precision(preds, labels, self.num_labels, average="weighted", top_k=k).item()
                for k in self.precision_at_k
            ],
            AssigneeMetrics.RECALL: [
                multiclass_recall(preds, labels, self.num_labels, top_k=k).item() for k in self.recall_at_k
            ],
            AssigneeMetrics.F1: [
                multiclass_f1_score(preds, labels, self.num_labels, top_k=k).item() for k in self.f1_at_k
            ],
            AssigneeMetrics.WAF1: [
                multiclass_f1_score(preds, labels, self.num_labels, average="weighted", top_k=k).item()
                for k in self.f1_at_k
            ],
        }

        return metrics
