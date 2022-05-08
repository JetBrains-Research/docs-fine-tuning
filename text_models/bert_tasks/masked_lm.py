import os.path

from sentence_transformers import models
from transformers import TrainingArguments, AutoModelForMaskedLM, AutoTokenizer, IntervalStrategy

from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.IREvalTrainer import IREvalTrainer
from text_models.datasets import BertModelMLMDataset

from data_processing.util import flatten


class MaskedLMTask(AbstractTask):
    def __init__(
        self, epochs=2, batch_size=16, eval_steps=200, n_examples="all", mask_probability=0.15, save_steps=5000
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, name="mlm")
        self.mask_probability = mask_probability
        self.save_steps = save_steps

    def finetune_on_docs(self, pretrained_model, docs_corpus, evaluator, max_len, device, save_to_path):
        corpus = AbstractTask.sections_to_sentences(docs_corpus)

        model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        inputs = tokenizer(corpus, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        dataset = BertModelMLMDataset(inputs, mask_probability=self.mask_probability, n_examples=self.n_examples)
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
        trainer.set_env_vars(evaluator, model, tokenizer, self.name, max_len, device)
        trainer.train()

        save_path = os.path.join(save_to_path, "mlm_pt_doc.model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return models.Transformer(save_path)
