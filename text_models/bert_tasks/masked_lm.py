import os.path

from sentence_transformers import models
from transformers import TrainingArguments, Trainer, BertConfig, BertForMaskedLM, AutoModelForMaskedLM, AutoTokenizer

from data_processing.util import get_corpus_properties
from text_models.bert_tasks import AbstractTask
from text_models.datasets import BertModelMLMDataset


class MaskedLMTask(AbstractTask):
    def __init__(self, epochs=2, batch_size=16, max_len=512, mask_probability=0.15, save_steps=5000):
        super().__init__(epochs, batch_size, max_len, name="mlm")
        self.mask_probability = mask_probability
        self.save_steps = save_steps

    def create_model_from_scratch(self, train_sentences, tmp_file):
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
        dumb_model = BertForMaskedLM(bert_config)

        return dumb_model, tokenizer

    def finetune_on_docs(self, pretrained_model, docs_corpus, device, save_to_path):

        model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        inputs = tokenizer(
            docs_corpus, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(inputs, mask_probability=self.mask_probability)
        args = TrainingArguments(
            output_dir=save_to_path,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_steps=self.save_steps,
            save_total_limit=3,
            disable_tqdm=False
        )
        trainer = Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()

        save_path = os.path.join(save_to_path, "mlm_pt_doc.model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        return models.Transformer(save_path)
