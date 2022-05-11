import os
from abc import abstractmethod
from typing import List

from sentence_transformers import evaluation, models
from transformers import AutoModelForNextSentencePrediction, AutoTokenizer, TrainingArguments, IntervalStrategy

from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.IREvalTrainer import IREvalTrainer


class SentencesClassificationTask(AbstractTask):
    def __init__(
        self,
        epochs=2,
        batch_size=16,
        eval_steps=200,
        n_examples="all",
        save_steps=2000,
        name="abstract_sentence_classification",
    ):
        super(SentencesClassificationTask, self).__init__(epochs, batch_size, eval_steps, n_examples, name)
        self.save_steps = save_steps

    def finetune_on_docs(
        self,
        pretrained_model: str,
        docs_corpus: List[List[List[str]]],
        evaluator: evaluation.InformationRetrievalEvaluator,
        max_len: int,
        device: str,
        save_to_path: str,
    ) -> models.Transformer:

        model = AutoModelForNextSentencePrediction.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        dataset = self._get_dataset(docs_corpus, tokenizer, max_len)
        args = TrainingArguments(
            output_dir=os.path.join(save_to_path, "checkpoints_docs"),
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_total_limit=3,
            disable_tqdm=False,
        )
        trainer = IREvalTrainer(model=model, args=args, train_dataset=dataset)
        trainer.set_env_vars(evaluator, model.bert, tokenizer, self.name, max_len, device)
        trainer.train()

        save_path = os.path.join(save_to_path, "nsp_pt_doc.model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return models.Transformer(save_path)

    @abstractmethod
    def _get_dataset(self, corpus, tokenizer, max_len):
        raise NotImplementedError()
