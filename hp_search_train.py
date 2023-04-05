import logging
import os

import pandas as pd
import torch
from omegaconf import OmegaConf

import wandb
from data_processing.util import load_config, get_corpus
from text_models import BertSiameseModel, TrainTypes
from text_models.bert_tasks.evaluation import ValMetric

hyperparameter_defaults = dict(
    learning_rate=2e-5,
    warmup_ratio=0.2,
    weight_decay=0.001,
    epochs=6,
    dataset="kotlin",
    load_path="text_models/saved/test",
    model_type=TrainTypes.PT_TASK,
)

siamese_config = load_config()


def train():

    wandb.init(config=hyperparameter_defaults)

    config = wandb.config

    datasets_config = OmegaConf.load(os.path.join("data", "datasets_config.yml"))[config["dataset"]]

    train_df = pd.read_csv(datasets_config.datasets.train)
    train_corpus_sent = get_corpus(train_df, sentences=True)
    disc_ids = train_df["disc_id"].tolist()

    logging.basicConfig(
        filename=os.path.join("data", "logs", "hp_" + siamese_config.dataset + "_" + config["model_type"] + ".txt"),
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    siamese_config.models.siamese["learning_rate"] = config["learning_rate"]
    siamese_config.models.siamese["save_to_path"] = config["load_path"]
    siamese_config.models.siamese["warmup_ratio"] = config["warmup_ratio"]
    siamese_config.models.siamese["weight_decay"] = config["weight_decay"]
    siamese_config.models.siamese["epochs"] = config["epochs"]
    siamese_config.models.siamese["finetuning_strategies"] = config["finetuning_strategies"]

    siamese_config.models.siamese["report_wandb"] = True
    siamese_config.models.siamese["hp_search_mode"] = True
    siamese_config.models.siamese["start_train_from_task"] = True
    
    model_type = config["model_type"]

    if model_type == TrainTypes.DOC_TASK or model_type == TrainTypes.BUGS_TASK:
        siamese_config.models.siamese["finetuning_strategies"] = ["mlm"]

    model = BertSiameseModel(train_corpus_sent, disc_ids, siamese_config.bert_tasks,
                             **siamese_config.models.siamese)

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
        model.train_pt_task(train_corpus_sent)
    elif model_type == TrainTypes.TASK:
        model.train_task(train_corpus_sent)

    wandb.log({"best_" + ValMetric.TASK: model.best_metric})

    torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
