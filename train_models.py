import argparse

import pandas as pd

from gensim.test.utils import get_tmpfile

from data_processing.util import get_corpus
from data_processing.util import get_docs_text

from models import W2VModel, FastTextModel, BertModelMLM, SBertModel


# for python <=3.7 support
class ExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.register("action", "my_extend", ExtendAction)
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument(
        "--docs", dest="docs", action="my_extend", nargs="+", help="Paths to preprocessed docs",
    )
    parser.add_argument(
        "-e", dest="epochs", action="store", type=int, help="The number of epochs for training saved_models",
    )
    parser.add_argument(
        "--vs", dest="vector_size", action="store", type=int, default=300, help="Embedding vector size",
    )
    parser.add_argument(
        "--min_count",
        dest="min_count",
        action="store",
        type=int,
        default=1,
        help="Ignore all words with total frequency lower than this in W2V, Fasttext and Random models",
    )
    parser.add_argument(
        "--tmp_file",
        dest="tmp_file",
        action="store",
        default=get_tmpfile("pretrained_vectors.txt"),
        help="The path to tmp file to store pretrained embeddings",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        action="store",
        type=int,
        default=16,
        help="Batch Size for Bert Model training.",
    )
    parser.add_argument("--w2v", dest="w2v", action="store_true", help="Train and save word2vec model")
    parser.add_argument("--fasttext", dest="fasttext", action="store_true", help="Train and save fasttext model")
    parser.add_argument("--bert", dest="bert", action="store_true", help="Train and save BERT model for MLM task.")
    parser.add_argument("--sbert", dest="sbert", action="store_true", help="Train and save SBERT model for STS tasks.")
    parser.add_argument(
        "--n_examples",
        dest="n_examples",
        action="store",
        type=int,
        help="Number of bug reports pairs for SBERT train for STS task",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    train = pd.read_csv(args.train)
    train_corpus = get_corpus(train)
    docs_corpus = get_docs_text(args.docs)

    if args.w2v:
        model = W2VModel(
            vector_size=args.vector_size, epochs=args.epochs, min_count=args.min_count, tmp_file=args.tmp_file
        )
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.fasttext:
        model = FastTextModel(vector_size=args.vector_size, epochs=args.epochs, min_count=args.min_count)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.bert:
        model = BertModelMLM(
            vector_size=args.vector_size, epochs=args.epochs, batch_size=args.batch_size, tmp_file=args.tmp_file
        )
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.sbert:
        if args.n_examples is None:
            raise ValueError("Please, specify n_examples parameter")

        disc_ids = train["disc_id"].tolist()
        model = SBertModel(
            train_corpus,
            disc_ids,
            vector_size=args.vector_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            tmp_file=args.tmp_file,
            n_examples=args.n_examples,
        )
        model.train_and_save_all(train_corpus, docs_corpus)


if __name__ == "__main__":
    main()
