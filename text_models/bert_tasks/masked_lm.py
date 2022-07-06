import os.path

from sentence_transformers import models, evaluation
from transformers import TrainingArguments, AutoModelForMaskedLM, AutoTokenizer, IntervalStrategy, AutoConfig

from data_processing.util import sections_to_sentences, Corpus
from text_models.bert_tasks import AbstractTask
from text_models.bert_tasks.eval_trainer import IREvalTrainer
from text_models.datasets import BertModelMLMDataset


class MaskedLMTask(AbstractTask):
    name = "mlm"

    def __init__(
        self,
        epochs: int = 2,
        batch_size: int = 16,
        eval_steps: int = 200,
        n_examples: int = "all",
        save_best_model: bool = False,
        mask_probability: float = 0.15,
        save_steps: int = 5000,
    ):
        super().__init__(epochs, batch_size, eval_steps, n_examples, save_best_model)
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

        checkpoints_path = os.path.join(save_to_path, "checkpoints_docs")
        output_path = os.path.join(save_to_path, "output_docs")

        args = TrainingArguments(
            output_dir=checkpoints_path,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=self.save_steps,
            save_total_limit=3,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.eval_steps,
            load_best_model_at_end=self.save_best_model,
            metric_for_best_model=f"MAP@{max(evaluator.map_at_k)}",
            greater_is_better=True,
            disable_tqdm=False,
        )

        trainer = IREvalTrainer(model=model, args=args, train_dataset=dataset)
        trainer.set_env_vars(evaluator, model, tokenizer, self.name, max_len, device)
        trainer.train()

        # if self.save_best_model == True we will use best model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        return models.Transformer(output_path)

    def load(self, load_from_path) -> models.Transformer:
        load_from_path = os.path.join(load_from_path, "output_docs")
        return models.Transformer(load_from_path)
