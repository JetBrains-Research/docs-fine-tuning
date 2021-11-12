import re
import string
import ast
import numpy as np

from pathlib import Path
from gensim.utils import simple_preprocess
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


def get_corpus(data):
    corpus = []
    for str_list in data["description"].tolist():
        word_lst = ast.literal_eval(str_list)
        corpus.append(word_lst)
    return corpus


def remove_noise(text):
    text = re.sub(r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", "", text, flags=re.MULTILINE)
    text = re.sub("\w*\d\w*", "", text)
    text = re.sub("\w*\f\w*", "", text)
    text = re.sub("\(.*?\)", "", text)
    text = re.sub("\[.*]\)", "", text)
    text = text.lower()
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)

    text = re.sub("[‘’“”…]", " ", text)
    text = re.sub("\n", " ", text)
    text = re.sub("\t", " ", text)

    return text


def lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos="v")


def tokenize_and_normalize(text):
    result = []
    STOPWORDS = stopwords.words("english") + ["http", "https"]
    for token in simple_preprocess(text, min_len=3):
        if token not in STOPWORDS:
            result.append(lemmatize(token))
    if len(result) == 0:
        return np.nan
    return result


def get_docs_text(docs_names):
    result = []
    for doc_name in docs_names:
        result.append(ast.literal_eval(Path(doc_name).read_text()))
    return result
