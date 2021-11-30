import os.path
import os

import numpy as np
import torch
import csv

from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import TrainingArguments, Trainer
from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer

from tqdm import tqdm

import gensim.downloader as api

from gensim.test.utils import get_tmpfile
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab

from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer

from data_processing.util import get_corpus_properties


class AbstractModel:
    def __init__(self, vector_size=300, epochs=5):
        self.vector_size = vector_size
        self.epochs = epochs
        self.name = "abstract"
        self.model = None

    def train_random(self, corpus):
        raise NotImplementedError()

    def train_pretrained(self, corpus):
        raise NotImplementedError()

    def train_finetuned(self, base_corpus, extra_corpus):
        self.train_pretrained(base_corpus + extra_corpus)

    def get_doc_embedding(self, doc):
        result = np.zeros(self.vector_size)
        for word in doc:
            result += self.model.wv[word]
        return result / len(doc)

    def get_embeddings(self, corpus, update_vocab=False):
        if update_vocab:
            self.model.build_vocab(corpus, update=True)

        embeddings = []
        for report in corpus:
            embeddings.append(self.get_doc_embedding(report))
        return np.array(embeddings)

    @staticmethod
    def load(path):
        raise NotImplementedError()

    def save(self, path):
        self.model.save(path)

    def train_and_save_all(self, base_corpus, extra_corpus):
        self.train_random(base_corpus)
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

    def train_random(self, corpus):
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

    def train_random(self, corpus):
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


# TODO: save embeddings format and build vocabulary
class GloveModel(AbstractModel):
    def __init__(self, corpus=None, vector_size=300, max_iter=100):
        super().__init__(vector_size)
        self.name = "glove"
        self.max_iter = max_iter
        self.vocab = self.__get_vocab(corpus)
        self.cooccurrence = self.__get_cooccurrance_matrix(corpus)
        self.pretrained = self.__glove2dict(os.path.join("pretrained_models", f"glove.6B.{self.vector_size}d.txt"))

        self.keyed_vectors = None

    def train_random(self, corpus):
        self.model = GloVe(n=self.vector_size, max_iter=self.max_iter)
        embeddings = self.model.fit(self.cooccurrence)
        self.keyed_vectors = self.__embeddings2model(embeddings)

    def train_pretrained(self, corpus):
        self.model = Mittens(n=self.vector_size, max_iter=self.max_iter)
        embeddings = self.model.fit(self.cooccurrence, vocab=self.vocab, initial_embedding_dict=self.pretrained)
        self.keyed_vectors = self.__embeddings2model(embeddings)

    def train_finetuned(self, base_corpus, extra_corpus):
        full_corpus = base_corpus + extra_corpus
        self.cooccurrence = self.__get_cooccurrance_matrix(full_corpus)
        self.vocab = self.__get_vocab(full_corpus)
        self.train_pretrained(full_corpus)

    def save(self, path):
        self.keyed_vectors.save(path)

    def load(self, path):
        self.model = KeyedVectors.load(path)

    def __get_cooccurrance_matrix(self, corpus):
        # TODO: get matrix with sliding window
        docs = [" ".join(doc) for doc in corpus]
        cv = CountVectorizer(ngram_range=(1, 1), vocabulary=self.vocab)
        X = cv.fit_transform(docs)
        Xc = X.T * X
        Xc.setdiag(0)
        return Xc.toarray()

    def __embeddings2model(self, embeddings):
        word_dict = dict(zip(self.vocab, embeddings))
        vocab_size = len(word_dict)
        result = KeyedVectors(self.vector_size)
        result.vectors = np.array([np.array(v) for v in word_dict.values()])

        for i, word in enumerate(word_dict.keys()):
            result.vocab[word] = Vocab(index=i, count=vocab_size - i)
            result.index2word.append(word)
        return result

    @staticmethod
    def __glove2dict(glove_filename):
        with open(glove_filename, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=" ", quoting=csv.QUOTE_NONE)
            embed = {line[0]: np.array(list(map(float, line[1:]))) for line in reader}
        return embed

    # TODO: OOV
    @staticmethod
    def __get_vocab(corpus):
        flatten_corpus = [token for doc in corpus for token in doc]
        return list(set(flatten_corpus))


class MeditationsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class BertModel(AbstractModel):
    def __init__(self, vector_size=384, epochs=5, batch_size=16, tmp_file=get_tmpfile("pretrained_vectors.txt")):
        super().__init__(vector_size, epochs)
        self.tokenizer = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.name = "bert"
        self.tmp_file = tmp_file
        self.batch_size = batch_size

        self.vocab_size = None
        self.max_len = 512

    def train_random(self, corpus):
        self.vocab_size, self.max_len = get_corpus_properties(corpus)
        self.max_len = 512

        train_sentences = [" ".join(doc) for doc in corpus]
        with open(self.tmp_file, "w") as fp:
            fp.write("\n".join(train_sentences))

        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[self.tmp_file],
            vocab_size=self.vocab_size,
            min_frequency=1,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        )

        tokenizer.save_model(os.path.join("pretrained_models", "bert_tokenizer"))

        self.tokenizer = RobertaTokenizer.from_pretrained(
            os.path.join("pretrained_models", "bert_tokenizer"), max_length=self.max_len
        )
        config = RobertaConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_len + 2,
            hidden_size=768,  # maybe vector_size
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        )
        self.model = RobertaForMaskedLM(config)

        inputs = self.tokenizer(
            train_sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModel.__get_dataset(inputs, mask_id=3, cls_id=0, sep_id=2, pad_id=1)
        self.__train(dataset)

    def train_pretrained(self, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModelForMaskedLM.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        sentences = [" ".join(sentence) for sentence in corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModel.__get_dataset(inputs)
        self.__train(dataset)

    def train_finetuned(self, base_corpus, extra_corpus):
        sentences = [" ".join(sentence) for sentence in extra_corpus]
        inputs = self.tokenizer(
            sentences, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        dataset = BertModel.__get_dataset(inputs)
        self.__train(dataset)

    @staticmethod
    def __get_dataset(inputs, mask_id=103, cls_id=102, sep_id=101, pad_id=0):
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

    def __train(self, dataset):
        args = TrainingArguments(
            output_dir="saved_models", per_device_train_batch_size=self.batch_size, num_train_epochs=self.epochs
        )

        trainer = Trainer(model=self.model, args=args, train_dataset=dataset)

        trainer.train()

    def get_doc_embedding(self, doc):
        return self.get_embeddings([doc])[0]

    def get_embeddings(self, corpus, update_vocab=False):
        descriptions = [" ".join(doc) for doc in corpus]  # full description embedding
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

            sentence_embeddings = BertModel.__mean_pooling(model_output, attention_mask)
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
        bertModel = BertModel()
        model = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        bertModel.model = model
        bertModel.device = device
        bertModel.tokenizer = tokenizer

        bertModel.model.to(device)
        return bertModel
