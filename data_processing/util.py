import ast
import re
from json import JSONEncoder
from pathlib import Path

import nltk
import numpy as np
from gensim.utils import simple_preprocess
from nltk import FreqDist
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from omegaconf import OmegaConf

CONFIG_PATH = "config.yml"


def get_corpus(data, sentences=False):
    corpus = []
    for str_list in data["description"].tolist():
        corpus.append(ast.literal_eval(str_list))
    return flatten(corpus) if sentences else list(map(flatten, corpus))


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_docs_text(docs_names, sections=False):
    result = []
    for doc_name in docs_names:
        text = Path(doc_name).read_text()
        result = result + get_doc_sections(text)
    return result if sections else flatten(result)

def get_doc_sections(text):
    sections = text.split(sep="]], [[")
    sections = [sections[0][1:] + "]]"] + ["[[" + section + "]]" for section in sections[1:-1]] + ["[[" + sections[-1][:-1]]
    sections = [get_doc_sentences(section) for section in sections]
    return sections

def get_doc_sentences(text):
    text = text.split(sep="], [")
    text = [sent.split("', '") for sent in text]
    for sent in text:
        sent[0] = sent[0][1:]
        sent[-1] = sent[-1][:-1]
    text[0][0] = text[0][0][2:]
    text[-1][-1] = text[-1][-1][:-2]
    return text


def remove_noise(text):
    text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", text, flags=re.MULTILINE)
    text = re.sub("\w*\d\w*", " ", text)
    text = re.sub("\w*\f\w*", " ", text)
    text = re.sub("\(.*?\)", " ", text)
    text = re.sub("\[.*]\)", " ", text)
    # remove non latin characters
    text = text.encode("ascii", "ignore")
    text = text.decode()
    text = text.lower()

    text = re.sub("[‘’“”…]", " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub("\t", " ", text)

    return text


def lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos="v")


def tokenize_and_normalize(sentences):
    result = []
    STOPWORDS = stopwords.words("english") + ["http", "https", "org", "use", "com"]
    for sentence in sentences:
        tokens = []
        for token in simple_preprocess(sentence, min_len=3):
            if token not in STOPWORDS:
                tokens.append(lemmatize(token))
        if len(tokens) >= 3:
            result.append(tokens)
    # in preprocess_csv.py we use DataFrame.dropna() method to
    if len(result) == 0:
        return np.nan
    return result


def parse_list(doc_name):
    return ast.literal_eval(Path(doc_name).read_text())


def replace_rarest_words(corpus, min_count):
    d = FreqDist()
    for docs in corpus:
        d.update(FreqDist(docs))

    for doc in corpus:
        for i in range(len(doc)):
            if d[doc[i]] < min_count:
                doc[i] = "<UNK>"
    return corpus


def get_corpus_properties(corpus):
    d = FreqDist()
    max_len = 0
    for docs in corpus:
        max_len = max(max_len, len(docs))
        d.update(FreqDist(docs))

    return len(d), max_len


def split_sentences(text):
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return list(filter(lambda x: len(x) > 3, tokenizer.tokenize(text)))


def load_config(path=None):
    cnf_path = CONFIG_PATH if path is None else path
    config = OmegaConf.load(cnf_path)
    for cnf in config.models.values():
        cnf["models_suffixes"] = config.models_suffixes
    return config


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
