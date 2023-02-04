import os.path
from typing import Union, Optional

from sentence_transformers import models, evaluation
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from datasets import Dataset

from data_processing.util import sections_to_sentences, Corpus
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.evaluation import ValMetric


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
        eval_steps: Optional[int] = None,
        n_examples: Union[int, str] = "all",
        val: float = 0.1,
        metric_for_best_model: str = ValMetric.TASK,
        save_best_model: bool = False,
        mask_probability: float = 0.15,
        save_steps: Optional[int] = None,
    ):
        super().__init__(
            epochs, batch_size, eval_steps, n_examples, val, metric_for_best_model, save_steps, save_best_model
        )
        self.mask_probability = mask_probability

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: Corpus,  # list of list(sections) of list(sentences) of tokens(words)
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
        report_wandb: bool = False,
    ) -> models.Transformer:
        corpus = sections_to_sentences(docs_corpus)

        config = AutoConfig.from_pretrained(pretrained_model)
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model, config=config)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=self.mask_probability)

        inputs = tokenizer(corpus, truncation=True)
        dataset = Dataset.from_dict(inputs).select(
            range(self.n_examples if self.n_examples != "all" and self.n_examples < len(corpus) else len(corpus))
        )
        val_task_dataset = Dataset.from_dict(tokenizer(evaluator.val_dataset, truncation=True))  # type: ignore

        return self._train_and_save(
            model,
            tokenizer,
            dataset,
            val_task_dataset,
            evaluator,
            save_to_path,
            self.save_steps,
            max_len,
            device,
            report_wandb,
            data_collator,
        )

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        return models.Transformer(load_from_path)
