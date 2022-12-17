import os.path
from typing import Union

from sentence_transformers import models, evaluation
from transformers import TrainingArguments, AutoModelForMaskedLM, AutoTokenizer, IntervalStrategy, AutoConfig

from data_processing.util import sections_to_sentences, Corpus
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.evaluation import IREvalTrainer
from text_models.datasets import BertModelMLMDataset


class MaskedLMTask(AbstractTask):
    """
    Masked Language Modeling task.

    :param epochs: Number of fine-tuning epochs
    :param batch_size: Batch size used for fine-tuning
    :param eval_steps: Number of update steps between two evaluations
    :param n_examples: Number of input examples that will be used for fine-tuning
    :param save_best_model: Whether or not to save the best model found during training at the end of training
    :param mask_probability: Probability that some word will be masked
    :param save_steps: Number of updates steps before two checkpoint saves
    """

    name = "mlm"

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: int = 200,
        n_examples: Union[int, str] = "all",
        val: float = 0.1,
        eval_with_task: bool = False,
        val_on_docs: bool = False,
        save_best_model: bool = False,
        mask_probability: float = 0.15,
        save_steps: int = 5000,
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, val, eval_with_task, val_on_docs, save_best_model)
        self.mask_probability = mask_probability
        self.save_steps = save_steps

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,  # list of list(sections) of list(sentences) of tokens(words)
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:
        corpus = sections_to_sentences(docs_corpus)

        config = AutoConfig.from_pretrained(pretrained_model)
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        inputs = tokenizer(corpus, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        dataset = BertModelMLMDataset(inputs, mask_probability=self.mask_probability, n_examples=self.n_examples)
        val_dataset_call = lambda: BertModelMLMDataset(
            tokenizer(evaluator.queries, max_length=max_len, padding="max_length",
                      truncation=True, return_tensors="pt"),
            mask_probability=self.mask_probability, n_examples=self.n_examples
        )

        return self._train_and_save(model, tokenizer, dataset, val_dataset_call, evaluator, save_to_path,
                                    self.save_steps, max_len, device)

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        return models.Transformer(load_from_path)
