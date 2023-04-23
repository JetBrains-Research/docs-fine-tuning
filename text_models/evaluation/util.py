import os

from torch.utils.data import Dataset


class ValMetric:
    LOSS_DOCS = "loss"
    LOSS_TASK = "loss_task"
    TASK = "task_map"


def write_csv_loss(loss_value: float, loss_task_value: float, output_path: str, epoch: float, steps: int):
    csv_path = os.path.join(output_path, "eval_loss.csv")
    if not os.path.isfile(csv_path):
        fOut = open(csv_path, mode="w", encoding="utf-8")
        fOut.write(",".join(["epoch", "step", "loss", "loss_task"]))
        fOut.write("\n")
    else:
        fOut = open(csv_path, mode="a", encoding="utf-8")

    output_data = [epoch, steps, loss_value, loss_task_value]
    fOut.write(",".join(map(str, output_data)))
    fOut.write("\n")
    fOut.close()

class ListDataset(Dataset):
    def __init__(self, data_list: list):
        self.data = data_list

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
