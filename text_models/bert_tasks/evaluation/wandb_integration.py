from typing import Dict, List

import wandb
from sentence_transformers.evaluation import SentenceEvaluator
from transformers.trainer_callback import TrainerState

from text_models.bert_tasks.evaluation.util import ValMetric


def define_metric(group_name, step_metric):
    wandb.define_metric(step_metric)
    wandb.define_metric(group_name + "*", step_metric=step_metric)


class WandbCallback:
    def __init__(self, task: str):
        self.step_metric = f"{task}/global_step"
        self.group_name = f"{task}/"

        define_metric(self.group_name, self.step_metric)

    def on_evaluate(self, state: TrainerState, metrics: Dict[str, float]):
        metrics = {self.group_name + metric_name: val for metric_name, val in metrics.items()}
        wandb.log({**metrics, self.step_metric: state.global_step})


class WandbLoggingEvaluator(SentenceEvaluator):
    def __init__(self, base_evaluator: SentenceEvaluator, step_metric: str, steps_per_epoch: int):
        self.base_evaluator = base_evaluator

        self.step_metric = step_metric
        self.group_name = step_metric.rsplit("/", 1)[0] + "/"

        self.steps_per_epoch = steps_per_epoch

        define_metric(self.group_name, step_metric)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        base_value = self.base_evaluator(model, output_path, epoch, steps)
        metrics = (
            {self.group_name + ValMetric.TASK: base_value}
            if self.base_evaluator.metrics is None
            else {self.group_name + metric_name: val for metric_name, val in self.base_evaluator.metrics.items()}
        )

        cur_step = epoch * self.steps_per_epoch + (steps if steps != -1 else self.steps_per_epoch)
        wandb.log({**metrics, self.step_metric: cur_step})

        return base_value
