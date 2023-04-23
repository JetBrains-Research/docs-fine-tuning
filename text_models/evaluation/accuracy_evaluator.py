from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
from sentence_transformers.util import batch_to_device
import os
import csv

logger = logging.getLogger(__name__)


class AccuracyEvaluator(SentenceEvaluator):

    def __init__(self, dataloader: DataLoader, write_csv: bool = True):
        self.dataloader = dataloader

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = "after epoch {}:".format(epoch)
            else:
                out_txt = "in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation " + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        accuracy = correct / total

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy
