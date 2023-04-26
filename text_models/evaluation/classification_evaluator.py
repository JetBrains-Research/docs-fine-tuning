import csv
import logging
import os
from typing import List, Optional, Dict

import torch
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import batch_to_device
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_precision, multiclass_recall

logger = logging.getLogger(__name__)


class AssigneeMetrics:
    ACC = "accuracy"
    WAF1 = "f1-score"
    WP = "precision"
    WR = "recall"


class AssignmentEvaluator(SentenceEvaluator):

    def __init__(self, dataset: Dataset, num_labels: int, val_corpus: Optional[List[str]] = None, batch_size=16,
                 write_csv: bool = True):
        self.dataset = dataset
        self.val_dataset = val_corpus
        self.batch_size = batch_size
        self.num_labels = num_labels

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", AssigneeMetrics.WAF1, AssigneeMetrics.WP,
                            AssigneeMetrics.WR, AssigneeMetrics.ACC]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = "after epoch {}:".format(epoch)
            else:
                out_txt = "in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation " + out_txt)
        metrics = self.compute_metrics(model)
        f1 = metrics[AssigneeMetrics.WAF1]
        precision = metrics[AssigneeMetrics.WP]
        recall = metrics[AssigneeMetrics.WR]
        accuracy = metrics[AssigneeMetrics.ACC]

        logger.info("Weighted average f1-score: {:.4f} \n".format(f1))
        logger.info("Weighted average precision: {:.4f} \n".format(precision))
        logger.info("Weighted average recall: {:.4f} \n".format(recall))
        logger.info("Accuracy: {:.4f} \n".format(accuracy))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, f1, precision, recall, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, f1, precision, recall, accuracy])

        return f1

    def compute_metrics(self, model) -> Dict[str, float]:
        model.eval()

        preds = []
        labels = []

        dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = model(features, labels=None)

            preds.append(torch.argmax(prediction, dim=1).to(model.device))
            labels.append(label_ids)

        preds = torch.cat(preds).to(model.device)
        labels = torch.cat(labels).to(model.device)

        metrics = {
            AssigneeMetrics.ACC: preds.eq(labels).mean().item(),
            AssigneeMetrics.WAF1: multiclass_f1_score(preds, labels, self.num_labels, average="weighted").item(),
            AssigneeMetrics.WP: multiclass_precision(preds, labels, self.num_labels, average="weighted").item(),
            AssigneeMetrics.WR: multiclass_recall(preds, labels, self.num_labels, average="weighted").item()
        }

        return metrics
