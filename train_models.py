import sys
import argparse
import os.path

import pandas as pd
import numpy as np

import gensim.downloader as api
from gensim.models.word2vec import Word2Vec

from gensim.test.utils import get_tmpfile

from data_processing.util import get_corpus
from data_processing.util import get_docs_text

from models import W2VModel, FastTextModel, GloveModel


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        dest="epochs",
        action="store",
        type=int,
        help="The number of epochs for training saved_models",
    )
    parser.add_argument(
        "--vs",
        dest="vector_size",
        action="store",
        type=int,
        default=300,
        help="Embedding vector size",
    )
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument(
        "--docs",
        dest="docs",
        action="extend",
        nargs="+",
        help="Paths to preprocessed docs",
    )
    parser.add_argument(
        "--tmp_file",
        dest="tmp_file",
        action="store",
        default=get_tmpfile("pretrained_vectors.txt"),
        help="The path to tmp file to store pretrained embeddings",
    )
    parser.add_argument("--w2v", dest="w2v", action="store_true", help="Train and save word2vec model")
    parser.add_argument("--fasttext", dest="fasttext", action="store_true", help="Train and save fasttext model")
    parser.add_argument("--glove", dest="glove", action="store_true", help="Train and save glove model")
    return parser.parse_args(arguments)


def get_init_vocab(vector_size, tmp_file):
    pretrained = api.load(f"glove-wiki-gigaword-{vector_size}")
    pretrained.save_word2vec_format(tmp_file)
    return [list(pretrained.key_to_index.keys())]


def train_random(train_corpus, vector_size, epochs):
    model = Word2Vec(train_corpus, vector_size=vector_size, min_count=0, epochs=epochs)
    model.save(os.path.join("saved_models", "word2vec_random.model"))


def train_pretrained(train_corpus, vector_size, epochs, init_vocab, tmp_file, is_finetuning=False):
    model = Word2Vec(vector_size=vector_size, min_count=0)
    model.build_vocab(train_corpus)

    model.build_vocab(init_vocab, update=True)
    model.wv.vectors_lockf = np.ones(len(model.wv))
    model.wv.intersect_word2vec_format(tmp_file, binary=False, lockf=1.0)

    model.train(train_corpus, total_examples=len(train_corpus), epochs=epochs)

    model_name = "word2vec_pretrained.model"
    if is_finetuning:
        model_name = "word2vec_finetuned.model"
    model.save(os.path.join("saved_models", model_name))


def main(args_str):
    args = parse_arguments(args_str)
    train = pd.read_csv(args.train)
    train_corpus = get_corpus(train)
    docs_corpus = get_docs_text(args.docs)

    if args.w2v:
        model = W2VModel(vector_size=args.vector_size, epochs=args.epochs, tmp_file=args.tmp_file)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.fasttext:
        model = FastTextModel(vector_size=args.vector_size, epochs=args.epochs)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.glove:
        model = GloveModel()
        model.train_and_save_all(train_corpus, docs_corpus)


if __name__ == "__main__":
    main(sys.argv[1:])
