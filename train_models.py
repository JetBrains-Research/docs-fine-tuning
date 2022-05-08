import argparse

import pandas as pd

from data_processing.util import get_corpus, get_docs_text, load_config
from text_models import W2VModel, FastTextModel, BertModelMLM, SBertModel, BertSiameseModel


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
    parser.add_argument(
        "--siamese",
        dest="siamese",
        action="store_true",
        help="Train and save SBERT model using siamese training approach",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()

    train = pd.read_csv(config.datasets.train)
    train_corpus = get_corpus(train)
    docs_corpus = get_docs_text(args.docs, sections=args.siamese)

    if args.w2v:
        model = W2VModel(**config.models.word2vec)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.fasttext:
        model = FastTextModel(**config.models.fasttext)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.bert:
        model = BertModelMLM(**config.models.bert)
        model.train_and_save_all(train_corpus, docs_corpus)

    disc_ids = train["disc_id"].tolist()
    if args.sbert:
        model = SBertModel(train_corpus, disc_ids, **config.models.sbert)
        model.train_and_save_all(train_corpus, docs_corpus)
    if args.siamese:
        model = BertSiameseModel(train_corpus, disc_ids, config.bert_tasks, **config.models.siamese)
        model.train_and_save_all(train_corpus, docs_corpus)


if __name__ == "__main__":
    main()
