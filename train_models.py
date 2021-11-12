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


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", dest="epochs", action="store", type=int, help="The number of epochs for training models",
    )
    parser.add_argument(
        "--vs", dest="vector_size", action="store", type=int, default=300, help="Embedding vector size",
    )
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument(
        "--docs", dest="docs", action="extend", nargs="+", help="Paths to preprocessed docs",
    )
    parser.add_argument(
        "--tmp_file",
        dest="tmp_file",
        action="store",
        default=get_tmpfile("pretrained_vectors.txt"),
        help="The path to tmp file to store pretrained embeddings",
    )
    return parser.parse_args(arguments)


def get_init_vocab(vector_size, tmp_file):
    pretrained = api.load(f"glove-wiki-gigaword-{vector_size}")
    pretrained.save_word2vec_format(tmp_file)
    return [list(pretrained.key_to_index.keys())]


def train_random(train_corpus, vector_size, epochs):
    model = Word2Vec(train_corpus, vector_size=vector_size, min_count=0, epochs=epochs)
    model.save(os.path.join("models", "word2vec_random.model"))


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
    model.save(os.path.join("models", model_name))


def main(args_str):
    args = parse_arguments(args_str)
    train = pd.read_csv(args.train)
    init_vocab = get_init_vocab(args.vector_size, args.tmp_file)
    train_corpus = get_corpus(train)
    docs_text = get_docs_text(args.docs)

    train_random(train_corpus, args.vector_size, args.epochs)
    print("Train random SUCCESS")
    train_pretrained(train_corpus, args.vector_size, args.epochs, init_vocab, args.tmp_file)
    print("Train pretrained SUCCESS")
    train_pretrained(
        train_corpus + docs_text, args.vector_size, args.epochs, init_vocab, args.tmp_file, is_finetuning=True,
    )
    print("Train fine-tuned SUCCESS")


if __name__ == "__main__":
    main(sys.argv[1:])
