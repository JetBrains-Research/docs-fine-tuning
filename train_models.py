import argparse
import logging
import os
import tempfile

import pandas as pd

from data_processing.util import get_corpus, get_docs_text, load_config
from text_models import W2VModel, FastTextModel, BertSiameseModel


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
    parser.add_argument(
        "--siamese",
        dest="siamese",
        action="store_true",
        help="Train and save SBERT model using siamese training approach",
    )

    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        action="store",
        type=str,
        default=None,
        help="GPU id for CUDA_VISIBLE_DEVICES environment param",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if config.tmpdir is not None:
        os.environ["TMPDIR"] = config.tmpdir
        tempfile.tempdir = config.tmpdir

    logging.basicConfig(
        filename=config.log_file, level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    train = pd.read_csv(config.datasets.train)
    train_corpus = get_corpus(train)
    docs_corpus = get_docs_text(args.docs, sections=True)

    if args.w2v:
        model = W2VModel(**config.models.word2vec)
        model.train_and_save_all(train_corpus, docs_corpus, config.model_types)
    if args.fasttext:
        model = FastTextModel(**config.models.fasttext)
        model.train_and_save_all(train_corpus, docs_corpus, config.model_types)

    disc_ids = train["disc_id"].tolist()
    if args.siamese:
        model = BertSiameseModel(train_corpus, disc_ids, config.bert_tasks, **config.models.siamese)
        model.train_and_save_all(train_corpus, docs_corpus, config.model_types)


if __name__ == "__main__":
    main()
