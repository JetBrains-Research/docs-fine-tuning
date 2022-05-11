from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.sentences_classification import SentencesClassificationTask
from text_models.datasets import NextSentenceDataset


class NextSentencePredictionTask(SentencesClassificationTask):
    def __init__(self, epochs=2, batch_size=16, eval_steps=200, n_examples="all", forget_const=10, save_steps=2000):
        super(NextSentencePredictionTask, self).__init__(
            epochs, batch_size, eval_steps, n_examples, save_steps, name="nsp"
        )
        self.forget_const = forget_const

    def _get_dataset(self, corpus, tokenizer, max_len):
        corpus = AbstractTask.sections_to_sentences(corpus)
        return NextSentenceDataset(corpus, tokenizer, self.n_examples, max_len, self.forget_const)
