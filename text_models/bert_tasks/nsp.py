import os
from typing import List

from sentence_transformers import models, evaluation
from transformers import AutoModelForNextSentencePrediction, AutoTokenizer, IntervalStrategy
from transformers import TrainingArguments

from text_models.bert_tasks import AbstractTask, IREvalTrainer
from text_models.bert_tasks.IREvalTrainer import IREvalTrainer
from text_models.datasets import NextSentenceDataset


class NextSentencePredictionTask(AbstractTask):
    def __init__(self, epochs=2, batch_size=16, eval_steps=200, n_examples="all", forget_const=10, save_steps=2000):
        super(NextSentencePredictionTask, self).__init__(epochs, batch_size, eval_steps, n_examples, name="nsp")
        self.forget_const = forget_const
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
        corpus = AbstractTask.sections_to_sentences(docs_corpus)

        model = AutoModelForNextSentencePrediction.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        dataset = NextSentenceDataset(corpus, tokenizer, self.n_examples, max_len, self.forget_const)
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
