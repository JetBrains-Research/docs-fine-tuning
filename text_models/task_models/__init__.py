from text_models.task_models.abstract_task import AbstractTask
from text_models.task_models.duplicate_detection import DuplicatesDetection
from text_models.task_models.assignment_recommendation import AssignmentRecommendationTask

finetuning_tasks = {
    DuplicatesDetection.name: DuplicatesDetection,
    AssignmentRecommendationTask.name: AssignmentRecommendationTask,
}
