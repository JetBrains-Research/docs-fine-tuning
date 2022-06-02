from text_models.bert_tasks.sentences_classification import SentencesClassificationTask
from text_models.datasets import SameSectionDataset


class SameSectionTask(SentencesClassificationTask):
    name = "sase"

    def __init__(
        self, epochs=2, batch_size=16, eval_steps=200, n_examples="all", save_best_model=False, save_steps=2000
    ):
        super(SameSectionTask, self).__init__(epochs, batch_size, eval_steps, n_examples, save_best_model, save_steps)

    def _get_dataset(self, corpus, tokenizer, max_len):
        sections = [[" ".join(sentence) for sentence in section] for section in corpus]
        return SameSectionDataset(sections, tokenizer, self.n_examples, max_len)
