import sys
import argparse

import pandas as pd

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
