import os


def write_csv_loss(loss_value: float, output_path: str, epoch: float, steps: int):
    csv_path = os.path.join(output_path, "eval_loss.csv")
    if not os.path.isfile(csv_path):
        fOut = open(csv_path, mode="w", encoding="utf-8")
        fOut.write(",".join(["epoch", "step", "loss"]))
        fOut.write("\n")
    else:
        fOut = open(csv_path, mode="a", encoding="utf-8")

    output_data = [epoch, steps, loss_value]
    fOut.write(",".join(map(str, output_data)))
    fOut.write("\n")
    fOut.close()