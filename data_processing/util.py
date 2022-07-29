import ast
import os.path
import re
from json import JSONEncoder
from pathlib import Path
from typing import List, Union, Any

import nltk
import numpy as np
import pandas as pd
import torch
from gensim.utils import simple_preprocess
from nltk import FreqDist
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from omegaconf import OmegaConf, DictConfig, ListConfig

CONFIG_PATH = "config.yml"

Sentence = List[str]
Section = List[Sentence]
Corpus = List[Section]
NumpyNaN = float


def get_corpus(data: pd.DataFrame, sentences: bool = False) -> Section:
    def parse_list(str_list):
        return [] if isinstance(str_list, float) else ast.literal_eval(str_list)

    corpus = []
    for summ, descr in zip(data.summary, data.description):
        full_descr = parse_list(summ) + parse_list(descr)
        corpus.append(full_descr)
    return flatten(corpus) if sentences else list(map(flatten, corpus))


def flatten(matrix: List[List[Any]]) -> List[Any]:
    return [item for sublist in matrix for item in sublist]


def get_docs_text(docs_names: List[str], sections: bool = False) -> Union[Section, Corpus]:
    result: Corpus = []
    for doc_name in docs_names:
        text = Path(doc_name).read_text()
        result = result + get_doc_sections(text)
    return result if sections else flatten(result)


def get_doc_sections(text: str) -> Corpus:
    sections = text.split(sep="]], [[")
    sections = (
        [sections[0][1:] + "]]"] + ["[[" + section + "]]" for section in sections[1:-1]] + ["[[" + sections[-1][:-1]]
    )
    return [get_doc_sentences(section) for section in sections]


def get_doc_sentences(text: str) -> Section:
    text_tmp_1 = text.split(sep="], [")
    text_tmp = [sent.split("', '") for sent in text_tmp_1]
    for sent in text_tmp:
        sent[0] = sent[0][1:]
        sent[-1] = sent[-1][:-1]
    text_tmp[0][0] = text_tmp[0][0][2:]
    text_tmp[-1][-1] = text_tmp[-1][-1][:-2]
    return text_tmp


def remove_noise(text: str) -> str:
    text = re.sub(r"(https|http)?://(\w|\.|/|\?|=|&|%)*\b", "", text, flags=re.MULTILINE)
    text = re.sub(r"\w*\d\w*", " ", text)
    text = re.sub(r"\w*\f\w*", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"\[.*]\)", " ", text)
    # remove non latin characters
    encoded_text = text.encode("ascii", "ignore")
    text = encoded_text.decode()
    text = text.lower()

    text = re.sub("[‘’“”…]", " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub("\t", " ", text)

    return text


def lemmatize(word: str) -> str:
    return WordNetLemmatizer().lemmatize(word, pos="v")


def tokenize_and_normalize(sentences: List[str]) -> Union[Section, NumpyNaN]:
    result = []
    eng_stopwords = stopwords.words("english") + ["http", "https", "org", "use", "com"]
    for sentence in sentences:
        tokens = []
        for token in simple_preprocess(sentence, min_len=3):
            if token not in eng_stopwords:
                tokens.append(lemmatize(token))
        if len(tokens) >= 3:
            result.append(tokens)
    # in preprocess_csv.py we use DataFrame.dropna() method to
    if len(result) == 0:
        return np.nan
    return result


def preprocess(text: str) -> Union[Section, NumpyNaN]:
    text = remove_noise(text)
    sentences = split_sentences(text)
    tokenized = tokenize_and_normalize(sentences)
    return tokenized


def sections_to_sentences(docs_corpus: Corpus) -> List[str]:
    return [" ".join(doc) for doc in flatten(docs_corpus)]


def get_corpus_properties(corpus: Section):
    freq_dict = FreqDist()
    max_len = 0
    for docs in corpus:
        max_len = max(max_len, len(docs))
        freq_dict.update(FreqDist(docs))

    return len(freq_dict), max_len


def split_sentences(text: str) -> List[str]:
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return list(filter(lambda x: len(x) > 3, tokenizer.tokenize(text)))


def randint_except(low: int, high: int, excluding: Union[List[int], np.ndarray]) -> int:
    result = np.random.randint(low, high - len(excluding))
    for ex in excluding:
        if result < ex:
            break
        result += 1
    return result


def fix_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_config(path: str = CONFIG_PATH) -> Union[ListConfig, DictConfig]:
    config = OmegaConf.load(path)
    datasets_config = OmegaConf.load(os.path.join("data", "datasets_config.yml"))[config.dataset]
    return OmegaConf.merge(config, datasets_config)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return JSONEncoder.default(self, o)
