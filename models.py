import os.path
import os

import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import BertModel, BertConfig, BertForMaskedLM, BertTokenizerFast
from tokenizers import BertWordPieceTokenizer

from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.readers import InputExample

from tqdm import tqdm

import gensim.downloader as api

from gensim.test.utils import get_tmpfile
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model

from data_processing.util import get_corpus_properties


class AbstractModel:
    def __init__(self, vector_size=300, epochs=5):
        self.vector_size = vector_size
        self.epochs = epochs
        self.name = "abstract"
        self.model = None

    def train_from_scratch(self, corpus):
        raise NotImplementedError()

    def train_pretrained(self, corpus):
        raise NotImplementedError()

    def train_finetuned(self, base_corpus, extra_corpus):
        self.train_pretrained(base_corpus + extra_corpus)

    def get_doc_embedding(self, doc):
        result = np.zeros(self.vector_size)
        size = 0
        for word in doc:
            if word in self.model.wv:
                result += self.model.wv[word]
                size += 1
        return result if size == 0 else result / size

    def get_embeddings(self, corpus, update_vocab=False):
        if update_vocab:
            self.model.build_vocab(corpus, update=True)

        return np.array([self.get_doc_embedding(report) for report in corpus]).astype(np.float32)

    @staticmethod
    def load(path):
        raise NotImplementedError()

    def save(self, path):
        self.model.save(path)

    def train_and_save_all(self, base_corpus, extra_corpus):
        self.train_from_scratch(base_corpus)
        print(f"Train random {self.name} SUCCESS")
        self.save(os.path.join("saved_models", f"{self.name}_random.model"))

        self.train_pretrained(base_corpus)
        print(f"Train pretrained {self.name} SUCCESS")
        self.save(os.path.join("saved_models", f"{self.name}_pretrained.model"))

        self.train_finetuned(base_corpus, extra_corpus)
        print(f"Train fine-tuned {self.name} SUCCESS")
        self.save(os.path.join("saved_models", f"{self.name}_finetuned.model"))


class W2VModel(AbstractModel):
    def __init__(self, vector_size=300, epochs=5, min_count=1, tmp_file=get_tmpfile("pretrained_vectors.txt")):
        super().__init__(vector_size, epochs)
        self.name = "w2v"
        self.tmp_file = tmp_file
        self.init_vocab = self.__get_init_vocab()
        self.min_count = min_count

    def train_from_scratch(self, corpus):
        self.model = Word2Vec(corpus, vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)

    def train_pretrained(self, corpus):
        if self.init_vocab is None:
            raise RuntimeError("Init vocab is None")

        self.model = Word2Vec(vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)
        self.model.build_vocab(corpus)
        self.model.min_count = 1
        self.model.build_vocab(self.init_vocab, update=True)
        self.model.wv.vectors_lockf = np.ones(len(self.model.wv))
        self.model.wv.intersect_word2vec_format(self.tmp_file, binary=False, lockf=1.0)
        self.model.min_count = self.min_count
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    def __get_init_vocab(self):
        pretrained = api.load(f"glove-wiki-gigaword-{self.vector_size}")  # TODO: change to 'word2vec-google-news-300'
        pretrained.save_word2vec_format(self.tmp_file)
        return [list(pretrained.key_to_index.keys())]

    @staticmethod
    def load(path):
        loaded_model = Word2Vec.load(path)
        created_model = W2VModel(loaded_model.vector_size, loaded_model.epochs, loaded_model.min_count)
        created_model.model = loaded_model
        return created_model


class FastTextModel(AbstractModel):
    def __init__(self, vector_size=300, epochs=5, min_count=1):
        super().__init__(vector_size, epochs)
        self.name = "ft"
        self.min_count = min_count

    def train_from_scratch(self, corpus):
        self.model = FastText(corpus, vector_size=self.vector_size, min_count=self.min_count, epochs=self.epochs)

    def train_pretrained(self, corpus):
        self.model = load_facebook_model(os.path.join("pretrained_models", "cc.en.300.bin"))
        self.model.build_vocab(corpus, update=True)
        self.model.train(corpus, total_examples=len(corpus), epochs=self.epochs)

    @staticmethod
    def load(path):
        loaded_model = FastText.load(path)
        created_model = FastTextModel(loaded_model.vector_size, loaded_model.epochs, loaded_model.min_count)
        created_model.model = loaded_model
        return created_model


class MeditationsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class BertModelMLM(AbstractModel):
    def __init__(self, vector_size=384, epochs=5, batch_size=16, tmp_file=get_tmpfile("pretrained_vectors.txt")):
        super().__init__(vector_size, epochs)
        self.tokenizer = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.name = "bert"
        self.tmp_file = tmp_file
        self.batch_size = batch_size

        self.vocab_size = None
        self.max_len = 512

    def train_from_scratch(self, corpus):
        train_sentences = [" ".join(doc) for doc in corpus]
        self.model, self.tokenizer = BertModelMLM.create_bert_model(train_sentences, self.tmp_file, self.max_len)

        inputs = self.tokenizer(
            train_sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLM.get_dataset(inputs, mask_id=4, cls_id=0, sep_id=2, pad_id=1)
        self.__train(dataset)

    def train_pretrained(self, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print(self.tokenizer(["I [MASK] love you", "When my time comes?"]))
        sentences = [" ".join(sentence) for sentence in corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLM.get_dataset(inputs)
        self.__train(dataset)

    def train_finetuned(self, base_corpus, extra_corpus):
        sentences = [" ".join(sentence) for sentence in extra_corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModelMLM.get_dataset(inputs)
        self.__train(dataset)

    @staticmethod
    def get_dataset(inputs, mask_id=103, cls_id=102, sep_id=101, pad_id=0):
        inputs["labels"] = inputs.input_ids.detach().clone()

        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (
            (rand < 0.15) * (inputs.input_ids != cls_id) * (inputs.input_ids != sep_id) * (inputs.input_ids != pad_id)
        )

        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = mask_id

        return MeditationsDataset(inputs)

    @staticmethod
    def create_bert_model(train_sentences, tmp_file, max_len, task="mlm"):
        if task == "mlm":
            model_type = BertForMaskedLM
        elif task == "sts":
            model_type = BertModel
        else:
            raise ValueError("Unsupported task")

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
        config = BertConfig(
            vocab_size=vocab_size,
            max_position_embeddings=max_len + 2,
            hidden_size=768,  # maybe vector_size
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        dumb_model = model_type(config)

        return dumb_model, tokenizer

    def __train(self, dataset):
        args = TrainingArguments(
            output_dir="saved_models", per_device_train_batch_size=self.batch_size, num_train_epochs=self.epochs
        )

        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)

        trainer.train()

    def get_doc_embedding(self, doc):
        return self.get_embeddings([doc])[0]

    def get_embeddings(self, corpus, update_vocab=False):
        descriptions = [" ".join(doc) for doc in corpus]
        encoded_input = self.tokenizer(
            descriptions, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = MeditationsDataset(encoded_input)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

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

    @staticmethod
    def load(path):
        bertModel = BertModelMLM()
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        bertModel.model = model
        bertModel.device = device
        bertModel.tokenizer = tokenizer

        bertModel.model.to(device)
        return bertModel


class SBertModel(AbstractModel):
    def __init__(
        self,
        corpus=None,
        disc_ids=None,
        vector_size=256,
        epochs=2,
        batch_size=16,
        tmp_file=get_tmpfile("pretrained_vectors.txt"),
        n_examples=None,
    ):
        super().__init__(vector_size, epochs)
        self.name = "sbert"
        self.tmp_file = tmp_file
        self.batch_size = batch_size

        if corpus is not None and disc_ids is not None:
            self.train_sts_dataloader = self.__get_train_dataloader_from_reports(corpus, disc_ids, n_examples)
            self.warmup_steps = int(len(self.train_sts_dataloader) * self.epochs * 0.1)  # 10% of train data

        self.max_len = 512

    def train_from_scratch(self, corpus):
        train_sentences = [" ".join(doc) for doc in corpus]
        dumb_model, tokenizer = BertModelMLM.create_bert_model(train_sentences, self.tmp_file, self.max_len, task="sts")

        dumb_model_name = os.path.join("saved_models", "dumb_bert")
        tokenizer.save_pretrained(dumb_model_name)
        dumb_model.save_pretrained(dumb_model_name)

        word_embedding_model = models.Transformer(dumb_model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )

        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=self.vector_size,
            activation_function=nn.Tanh(),
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        self.__train_sts(self.train_sts_dataloader)

    def train_pretrained(self, corpus):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.__train_sts(self.train_sts_dataloader)

    def train_finetuned(self, base_corpus, extra_corpus):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        extra_train_dataloader = self.__get_train_dataloader_from_docs(extra_corpus)
        self.__train_sts(extra_train_dataloader)
        self.__train_sts(self.train_sts_dataloader)

    def __train_sts(self, train_dataloader):
        train_loss = losses.CosineSimilarityLoss(self.model)
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)], epochs=self.epochs, warmup_steps=self.warmup_steps
        )

    def get_embeddings(self, corpus, update_vocab=False):
        return self.model.encode([" ".join(report) for report in corpus]).astype(np.float32)

    def get_doc_embedding(self, doc):
        return self.get_embeddings([" ".join(doc)])[0]

    @staticmethod
    def load(path):
        sbert_model = SentenceTransformer(path)
        model = SBertModel()
        model.model = sbert_model
        return model

    def save(self, path):
        self.model.save(path)

    def __get_train_dataloader_from_reports(self, corpus, disc_ids, n_examples):
        corpus = list(map(lambda x: " ".join(x), corpus))
        train_data = []
        lngth = len(corpus)
        for i in range(lngth):
            for j in range(i + 1, lngth):
                label = 1.0 if disc_ids[i] == disc_ids[j] else 0.0
                train_data.append(InputExample(texts=[corpus[i], corpus[j]], label=label))
        np.random.shuffle(train_data)
        if n_examples is not None:
            train_data = train_data[:n_examples]
        return DataLoader(train_data, shuffle=True, batch_size=self.batch_size)

    def __get_train_dataloader_from_docs(self, docs_corpus):
        train_data = []
        corpus = list(map(lambda x: " ".join(x), docs_corpus))
        lngth = len(docs_corpus) - 1
        forget_const = 10
        for i in range(lngth):
            train_data.append(InputExample(texts=[corpus[i], corpus[i + 1]], label=1.0))
            if i + forget_const < lngth:
                train_data.append(
                    InputExample(texts=[corpus[i], corpus[i + np.random.randint(forget_const, lngth - i)]], label=0.0)
                )

        np.random.shuffle(train_data)
        return DataLoader(train_data, shuffle=True, batch_size=self.batch_size)
