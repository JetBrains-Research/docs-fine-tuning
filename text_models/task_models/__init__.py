from text_models.task_models.abstract_task import AbstractTask
from text_models.task_models.duplicate_detection import DuplicatesDetection

finetuning_tasks = {
    DuplicatesDetection.name: DuplicatesDetection
}