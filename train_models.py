import argparse
import logging
import os
import tempfile

from datetime import datetime
import pandas as pd

from data_processing.util import flatten, get_corpus, get_docs_text, load_config
from text_models import W2VModel, FastTextModel, BertSiameseModel


def parse_arguments():
    parser = argparse.ArgumentParser()

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
    train_corpus_sent = get_corpus(train, sentences=True)
    train_corpus = list(map(flatten, train_corpus_sent))
    docs_corpus = get_docs_text(config.docs, sections=True)

    if config.text_model == "word2vec":
        model = W2VModel(**config.models.word2vec)
        model.train_and_save_all(train_corpus, docs_corpus, config.model_types)
    if config.text_model == "fasttext":
        model = FastTextModel(**config.models.fasttext)
        model.train_and_save_all(train_corpus, docs_corpus, config.model_types)

    if config.text_model == "siamese":
        os.environ["WANDB_RUN_GROUP"] = config.dataset + "-so-" + datetime.now().strftime("%d-%m-%yT%H:%M:%S")
        disc_ids = train["disc_id"].tolist()
        model = BertSiameseModel(train_corpus_sent, disc_ids, config.bert_tasks, **config.models.siamese)
        model.train_and_save_all(train_corpus, docs_corpus, config.model_types)


if __name__ == "__main__":
    main()
