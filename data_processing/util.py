import re
import string
import ast
import numpy as np

from pathlib import Path
from gensim.utils import simple_preprocess
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist


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
    STOPWORDS = stopwords.words("english") + ["http", "https", "org", "use", "com"]
    for token in simple_preprocess(text, min_len=3):
        if token not in STOPWORDS:
            result.append(lemmatize(token))
    if len(result) == 0:
        return np.nan
    return result


def get_docs_text(docs_names):
    result = []
    for doc_name in docs_names:
        result = result + get_doc_sentences(doc_name)
    return result


def get_doc_sentences(doc_name):
    text = Path(doc_name).read_text()
    text = text.split(sep="], [")
    text = [sent.split("', '") for sent in text]
    for sent in text:
        sent[0] = sent[0][1:]
        sent[-1] = sent[-1][:-1]
    text[0][0] = text[0][0][2:]
    text[-1][-1] = text[-1][-1][:-2]
    return text


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
