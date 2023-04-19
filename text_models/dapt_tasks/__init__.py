from text_models.dapt_tasks.abstract_task import AbstractPreTrainingTask
from text_models.dapt_tasks.nsp import NextSentencePredictionTask
from text_models.dapt_tasks.same_section_task import SameSectionTask
from text_models.dapt_tasks.masked_lm import MaskedLMTask
from text_models.dapt_tasks.sts import STSTask
from text_models.dapt_tasks.tsdae import TSDenoisingAutoEncoderTask

tasks = {
    MaskedLMTask.name: MaskedLMTask,
    STSTask.name: STSTask,
    NextSentencePredictionTask.name: NextSentencePredictionTask,
    TSDenoisingAutoEncoderTask.name: TSDenoisingAutoEncoderTask,
    SameSectionTask.name: SameSectionTask,
}
