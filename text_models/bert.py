import os.path
import os

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import BertModel, BertConfig, BertForMaskedLM, BertTokenizerFast
from tokenizers import BertWordPieceTokenizer

from tqdm import tqdm

from gensim.test.utils import get_tmpfile

from data_processing.util import get_corpus_properties
from text_models.abstract_model import AbstractModel


class BertModelDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class BertModelMLMDataset(BertModelDataset):
    def __init__(self, encodings, mask_id=103, cls_id=102, sep_id=101, pad_id=0, mask_probability=0.15):
        super(BertModelMLMDataset, self).__init__(encodings)

        self.encodings["labels"] = self.encodings.input_ids.detach().clone()
        self.mask_proba = mask_probability
        self.mask_id = mask_id
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id

    def __getitem__(self, idx):
        inputs = self.encodings.input_ids[idx]

        rand = torch.rand(inputs.shape)
        mask_arr = (
            (rand < self.mask_proba) * (inputs != self.cls_id) * (inputs != self.sep_id) * (inputs != self.pad_id)
        )
        inputs[mask_arr] = self.mask_id

        self.encodings.input_ids[idx] = inputs
        return super().__getitem__(idx)


class BertModelMLM(AbstractModel):
    def __init__(
        self,
        vector_size=384,
        epochs=5,
        batch_size=16,
        pretrained_model="bert-base-uncased",
        tmp_file=get_tmpfile("pretrained_vectors.txt"),
    ):
        super().__init__(vector_size, epochs, pretrained_model)
        self.tokenizer = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.tmp_file = tmp_file
        self.batch_size = batch_size

        self.vocab_size = None
        self.max_len = 512

    name = "bert"

    def train_from_scratch(self, corpus):
        train_sentences = [" ".join(doc) for doc in corpus]
        self.model, self.tokenizer = BertModelMLM.create_bert_model(train_sentences, self.tmp_file, self.max_len)

        inputs = self.tokenizer(
            train_sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(inputs, mask_id=4, cls_id=0, sep_id=2, pad_id=1)
        self.__train(dataset)

    def train_pretrained(self, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.model = AutoModelForMaskedLM.from_pretrained(self.pretrained_model)
        print(self.tokenizer(["I [MASK] love you", "When my time comes?"]))
        sentences = [" ".join(sentence) for sentence in corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(inputs)
        self.__train(dataset)

    def train_finetuned(self, base_corpus, extra_corpus):
        sentences = [" ".join(sentence) for sentence in extra_corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLMDataset(inputs)
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

        tokenizer.save_model(os.path.join("pretrained_models", "bert_tokenizer"))

        tokenizer = BertTokenizerFast.from_pretrained(
            os.path.join("pretrained_models", "bert_tokenizer"), max_len=max_len + 2
        )
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
            output_dir="saved_models", per_device_train_batch_size=self.batch_size, num_train_epochs=self.epochs
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
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        bertModel.model = model
        bertModel.device = device
        bertModel.tokenizer = tokenizer

        bertModel.model.to(device)
        return bertModel
