import logging
import os

import pandas as pd
import torch
from omegaconf import OmegaConf

import wandb
from data_processing.util import load_config, fix_random_seed
from text_models import BertDomainModel, TrainTypes
from text_models.task_models import finetuning_tasks

hyperparameter_defaults = dict(
    learning_rate=2e-5,
    warmup_ratio=0.2,
    weight_decay=0.001,
    epochs=6,
    dataset="kotlin",
    load_path="text_models/saved/test",
    model_type=TrainTypes.PT_TASK,
    task="duplicates_detection",
)

config = load_config()


def train():
    wandb.init(config=hyperparameter_defaults)

    fix_random_seed()

    wandb_config = wandb.config
    datasets_config = OmegaConf.load(os.path.join("data", "datasets_config.yml"))[wandb_config["dataset"]]
    task_config = config.target_tasks[config.target_task]
    logging.basicConfig(
        filename=os.path.join("data", "logs", "hp_" + config.dataset + "_" + wandb_config["model_type"] + ".txt"),
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    train_df = pd.read_csv(datasets_config.datasets.train)

    task_config["learning_rate"] = wandb_config["learning_rate"]
    task_config["warmup_ratio"] = wandb_config["warmup_ratio"]
    task_config["weight_decay"] = wandb_config["weight_decay"]
    task_config["epochs"] = wandb_config["epochs"]
    task_config["dropout_ratio"] = wandb_config["dropout_ratio"]
    task_config["batch_size"] = wandb_config["batch_size"]


    config.models.bert["domain_adaptation_tasks"] = wandb_config["domain_adaptation_tasks"]
    config.models.bert["save_to_path"] = wandb_config["load_path"]
    config.models.bert["report_wandb"] = True
    config.models.bert["hp_search_mode"] = True
    config.models.bert["start_train_from"] = "task"

    model_type = wandb_config["model_type"]

    if model_type == TrainTypes.DOC_TASK or model_type == TrainTypes.BUGS_TASK:
        config.models.bert["domain_adaptation_tasks"] = ["mlm"]

    target_task = finetuning_tasks[config.target_task].load(train_df, task_config)
    model = BertDomainModel(target_task, config.dapt_tasks, **config.models.bert)

    if model_type == TrainTypes.DOC_TASK:
        model.train_doc_task([], [])
    elif model_type == TrainTypes.PT_DOC_TASK:
        model.train_pt_doc_task([], [])
    elif model_type == TrainTypes.BUGS_TASK:
        model.train_bugs_task()
    elif model_type == TrainTypes.PT_BUGS_TASK:
        model.train_pt_bugs_task()
    elif model_type == TrainTypes.DOC_BUGS_TASK:
        model.train_doc_bugs_task([])
    elif model_type == TrainTypes.PT_DOC_BUGS_TASK:
        model.train_pt_doc_bugs_task([])
    elif model_type == TrainTypes.PT_TASK:
        model.train_pt_task([])
    elif model_type == TrainTypes.TASK:
        model.train_task([])

    wandb.log({"best_w_f1": model.best_metric})

    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
