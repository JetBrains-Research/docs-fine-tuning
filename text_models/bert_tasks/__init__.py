from text_models.bert_tasks.abstract_task import AbstractTask
from text_models.bert_tasks.masked_lm import MaskedLMTask
from text_models.bert_tasks.nsp import NextSentencePredictionTask
from text_models.bert_tasks.same_section_task import SameSectionTask
from text_models.bert_tasks.sts import STSTask
from text_models.bert_tasks.tsdae import TSDenoisingAutoEncoderTask

tasks = {
    "mlm": MaskedLMTask,
    "sts": STSTask,
    "nsp": NextSentencePredictionTask,
    "tsdae": TSDenoisingAutoEncoderTask,
    "same_section": SameSectionTask,
}
