import sys
import argparse

import pandas as pd
import numpy as np

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

from gensim.test.utils import get_tmpfile

from data_processing.util import get_corpus
from data_processing.util import get_docs_text


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='epochs', action='store', type=int)
    parser.add_argument('--vs', dest='vector_size', action='store', type=int)
    parser.add_argument('--train', dest='train', action='store')
    parser.add_argument('--docs', dest='docs', action='extend', nargs="+")
    return parser.parse_args(arguments)


def get_init_vocab(vector_size):
    pretrained = api.load(f"glove-wiki-gigaword-{vector_size}")
    pretrained.save_word2vec_format(tmp_file)
    return [list(pretrained.key_to_index.keys())]


def train_random(train_corpus, vector_size, epochs):
    model = Word2Vec(train_corpus, vector_size=vector_size, min_count=0, epochs=epochs)
    model.save('models/word2vec_random.model')


def pretrain_model(model, init_vocab):
    model.build_vocab(init_vocab, update=True)
    model.wv.vectors_lockf = np.ones(len(model.wv))
    model.wv.intersect_word2vec_format(tmp_file, binary=False, lockf=1.0)
    return model


def train_pretrained(train_corpus, vector_size, epochs, init_vocab):
    model = Word2Vec(vector_size=vector_size, min_count=0)
    model.build_vocab(train_corpus)
    model = pretrain_model(model, init_vocab)
    model.train(train_corpus, total_examples=len(train_corpus), epochs=epochs)
    model.save('models/word2vec_pretrained.model')


def train_finetuned(train_corpus, vector_size, epochs, docs, init_vocab):
    model = Word2Vec(vector_size=vector_size, min_count=0)
    model.build_vocab(train_corpus)
    model.build_vocab(docs, update=True)
    model = pretrain_model(model, init_vocab)
    model.train(docs, total_examples=len(docs), epochs=epochs)
    model.train(train_corpus, total_examples=len(train_corpus), epochs=epochs)
    model.save('models/word2vec_finetuned.model')


def main(args_str):
    args = parse_arguments(args_str)
    train = pd.read_csv(args.train)
    init_vocab = get_init_vocab(args.vector_size)
    train_corpus = get_corpus(train)
    docs = get_docs_text(args.docs)

    train_random(train_corpus, args.vector_size, args.epochs)
    print("Train random SUCCESS")
    train_pretrained(train_corpus, args.vector_size, args.epochs, init_vocab)
    print("Train pretrained SUCCESS")
    train_finetuned(train_corpus, args.vector_size, args.epochs, docs, init_vocab)
    print("Train finetuned SUCCESS")


if __name__ == '__main__':
    tmp_file = get_tmpfile("pretrained_vectors.txt")
    main(sys.argv[1:])





