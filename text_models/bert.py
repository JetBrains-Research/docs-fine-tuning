import tempfile

import numpy as np
import torch
from deprecated import deprecated
from gensim.test.utils import get_tmpfile
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import BertModel, BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import TrainingArguments, Trainer

from data_processing.util import get_corpus_properties
from text_models import AbstractModel
from text_models.datasets import BertModelDataset, BertModelMLMDataset


@deprecated("Will be removed in the future. Use BertSiameseModel with MLM task instead")
class BertModelMLM(AbstractModel):
    def __init__(
        self,
        vector_size=384,
        epochs=5,
        batch_size=16,
        mask_probability=0.15,
        max_len=512,
        pretrained_model="bert-base-uncased",
        tmp_file=get_tmpfile("pretrained_vectors.txt"),
        device="cpu",
        seed=42,
        save_to_path="./",
    ):
        super().__init__(vector_size, epochs, pretrained_model, seed, save_to_path)
        self.tokenizer = None
        self.device = torch.device(device)
        self.tmp_file = tmp_file or get_tmpfile("pretrained_vectors.txt")
        self.batch_size = batch_size
        self.mask_probability = mask_probability

        self.vocab_size = None
        self.max_len = max_len

    name = "BERT"

    def train_task(self, corpus):
        train_sentences = [" ".join(doc) for doc in corpus]
        self.model, self.tokenizer = BertModelMLM.create_bert_model(train_sentences, self.tmp_file, self.max_len)

        inputs = self.tokenizer(
            train_sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(
            inputs, mask_id=4, cls_id=0, sep_id=2, pad_id=1, mask_probability=self.mask_probability
        )
        self.__train(dataset)

    def train_pt_task(self, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model)
        sentences = [" ".join(sentence) for sentence in corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(inputs, mask_probability=self.mask_probability)
        self.__train(dataset)

    def train_doc_task(self, base_corpus, extra_corpus):
        raise NotImplementedError()

    def train_pt_doc_task(self, base_corpus, extra_corpus):
        sentences = [" ".join(sentence) for sentence in extra_corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(inputs, mask_probability=self.mask_probability)
        self.__train(dataset)

    @staticmethod
    def get_model_type_by_task(task):
        if task == "mlm":
            return BertForMaskedLM
        if task == "sts":
            return BertModel
        raise ValueError("Unsupported task")

    @staticmethod
    def create_bert_model(train_sentences, tmp_file, max_len, task="mlm"):
        model_type = BertModelMLM.get_model_type_by_task(task)

        vocab_size, _ = get_corpus_properties([sentence.split(" ") for sentence in train_sentences])
        with open(tmp_file, "w") as fp:
            fp.write("\n".join(train_sentences))

        tokenizer = BertWordPieceTokenizer(
            clean_text=True, handle_chinese_chars=False, strip_accents=False, lowercase=True
        )
        tokenizer.train(
            files=[tmp_file],
            vocab_size=vocab_size,
            min_frequency=1,
            wordpieces_prefix="##",
            special_tokens=["[CLS]", "[PAD]", "[SEP]", "[UNK]", "[MASK]"],
        )

        save_to_path = tempfile.mkdtemp()

        tokenizer.save_model(save_to_path)

        tokenizer = BertTokenizerFast.from_pretrained(save_to_path, max_len=max_len + 2)
        bert_config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_len + 2,
            hidden_size=768,  # maybe vector_size
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        dumb_model = model_type(bert_config)

        return dumb_model, tokenizer

    def __train(self, dataset):
        args = TrainingArguments(
            output_dir=self.save_to_path, per_device_train_batch_size=self.batch_size, num_train_epochs=self.epochs
        )

        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)

        trainer.train()

    def get_doc_embedding(self, doc):
        return self.get_embeddings([doc])[0]

    def get_embeddings(self, corpus):
        descriptions = [" ".join(doc) for doc in corpus]
        encoded_input = self.tokenizer(
            descriptions, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelDataset(encoded_input)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        result = []
        loop = tqdm(loader, leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            with torch.no_grad():
                model_output = self.model(input_ids, attention_mask=attention_mask)

            sentence_embeddings = BertModelMLM.__mean_pooling(model_output, attention_mask)
            result += sentence_embeddings.tolist()

        return np.array(result).astype(np.float32)

    @staticmethod
    def __mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path):
        bertModel = BertModelMLM()
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # TODO

        bertModel.model = model
        bertModel.device = device
        bertModel.tokenizer = tokenizer

        bertModel.model.to(device)
        return bertModel
