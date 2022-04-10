import argparse

import pandas as pd

from gensim.test.utils import get_tmpfile
from omegaconf import OmegaConf

from data_processing.util import get_corpus, get_docs_text
from data_processing.util import CONFIG_PATH

from text_models import W2VModel, FastTextModel, BertModelMLM, SBertModel

# for python <=3.7 support
class ExtendAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.register("action", "my_extend", ExtendAction)
    parser.add_argument(
        "--docs",
        dest="docs",
        action="my_extend",
        nargs="+",
        help="Paths to preprocessed docs",
    )
    parser.add_argument("--w2v", dest="w2v", action="store_true", help="Train and save word2vec model")
    parser.add_argument("--fasttext", dest="fasttext", action="store_true", help="Train and save fasttext model")
    parser.add_argument("--bert", dest="bert", action="store_true", help="Train and save BERT model for MLM task.")
    parser.add_argument("--sbert", dest="sbert", action="store_true", help="Train and save SBERT model for STS tasks.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = OmegaConf.load(CONFIG_PATH)

    train = pd.read_csv(config.datasets.train)
    train_corpus = get_corpus(train)
    docs_corpus = get_docs_text(args.docs)

    if args.w2v:
        cnf_w2v = config.models.w2v
        model = W2VModel(
            vector_size=cnf_w2v.vector_size,
            epochs=cnf_w2v.epochs,
            min_count=cnf_w2v.min_count,
            tmp_file=cnf_w2v.tmp_file or get_tmpfile("pretrained_vectors.txt"),
        )
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.fasttext:
        cnf_ft = config.models.fasttext
        model = FastTextModel(vector_size=cnf_ft.vector_size, epochs=cnf_ft.epochs, min_count=cnf_ft.min_count)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.bert:
        cnf_bert = config.models.bert
        model = BertModelMLM(
            vector_size=cnf_bert.vector_size,
            epochs=cnf_bert.epochs,
            batch_size=cnf_bert.batch_size,
            tmp_file=cnf_bert.tmp_file or get_tmpfile("pretrained_vectors.txt"),
        )
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.sbert:
        cnf_sbert = config.models.sbert
        disc_ids = train["disc_id"].tolist()
        model = SBertModel(
            train_corpus,
            disc_ids,
            vector_size=cnf_sbert.vector_size,
            epochs=cnf_sbert.epochs,
            batch_size=cnf_sbert.batch_size,
            tmp_file=cnf_sbert.tmp_file or get_tmpfile("pretrained_vectors.txt"),
            n_examples=cnf_sbert.n_examples,
        )
        model.train_and_save_all(train_corpus, docs_corpus)


if __name__ == "__main__":
    main()
