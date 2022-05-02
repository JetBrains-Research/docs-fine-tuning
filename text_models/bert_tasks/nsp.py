import os
from typing import List

from sentence_transformers import models
from transformers import AutoModelForNextSentencePrediction, AutoTokenizer
from transformers import BertConfig, BertForNextSentencePrediction
from transformers import TrainingArguments, Trainer

from data_processing.util import get_corpus_properties
from text_models.bert_tasks import AbstractTask
from text_models.datasets import NextSentenceDataset


class NextSentencePredictionTask(AbstractTask):
    def __init__(self, epochs=2, batch_size=16, n_examples="all", max_len=512, forget_const=10, save_steps=2000):
        super(NextSentencePredictionTask, self).__init__(epochs, batch_size, n_examples, max_len, name="nsp")
        self.forget_const = forget_const
        self.save_steps = save_steps

    def create_model_from_scratch(self, train_sentences: List[str], tmp_file: str):
        vocab_size, _ = get_corpus_properties([sentence.split(" ") for sentence in train_sentences])

        tokenizer = self._create_and_train_tokenizer(train_sentences, vocab_size, tmp_file)

        bert_config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=self.max_len + 2,
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        dumb_model = BertForNextSentencePrediction(bert_config)

        return dumb_model, tokenizer

    def finetune_on_docs(
        self, pretrained_model: str, docs_corpus: List[str], device: str, save_to_path: str
    ) -> models.Transformer:
        model = AutoModelForNextSentencePrediction.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        dataset = NextSentenceDataset(docs_corpus, tokenizer, self.n_examples, self.max_len, self.forget_const)
        args = TrainingArguments(
            output_dir=save_to_path,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_steps=self.save_steps,
            save_total_limit=3,
            disable_tqdm=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        save_path = os.path.join(save_to_path, "mlm_pt_doc.model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return models.Transformer(save_path)
