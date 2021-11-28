import sys
import argparse
import numpy as np
import pandas as pd

from models import AbstractModel, W2VModel, FastTextModel
from data_processing.util import get_corpus

import faiss


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument("--test", dest="test", action="store", help="The path to test dataset")
    parser.add_argument("-r", dest="model_random", action="store", help="The path to random model")
    parser.add_argument(
        "-p",
        dest="model_pretrained",
        action="store",
        help="The path to pretrained model",
    )
    parser.add_argument(
        "-f",
        dest="model_finetuned",
        action="store",
        help="The path to fine-tuned model",
    )
    parser.add_argument(
        "--topn",
        dest="topn",
        action="store",
        type=int,
        default=5,
        help="The number of predicted duplicate bug-reports for one report",
    )
    parser.add_argument("--w2v", dest="w2v", action="store_true", help="Use word2vec model for classification")
    parser.add_argument(
        "--fasttext", dest="fasttext", action="store_true", help="Use fasttext model for classification"
    )
    return parser.parse_args(arguments)


def get_recall(train: pd.DataFrame, test: pd.DataFrame, model: AbstractModel, topn: int):
    index = faiss.IndexFlatIP(model.vector_size)

    train_corpus = get_corpus(train)
    embeddings = model.get_embeddings(train_corpus).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    test_corpus = get_corpus(test)
    test_embs = model.get_embeddings(test_corpus, update_vocab=True).astype(np.float32)

    test_size = 0
    TP = 0
    for ind, descr in enumerate(test_corpus):
        if test.iloc[ind]["id"] != test.iloc[ind]["disc_id"]:  # not in master_ids
            dupl_ids = index.search(test_embs[ind].reshape(1, -1), topn)[1][0]
            val = 0
            for dupl_id in dupl_ids:
                if train.iloc[dupl_id]["disc_id"] == test.iloc[ind]["disc_id"]:
                    val = 1
                    break
            TP += val
            test_size += 1

        train = train.append(test.iloc[ind], ignore_index=True)

        tmp_emb = np.array([test_embs[ind]])
        faiss.normalize_L2(tmp_emb)
        index.add(tmp_emb)

    return TP / test_size


def main(args_str):
    args = parse_arguments(args_str)
    if args.w2v is None and args.fasttext is None:
        raise ValueError("Please select a model")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    model_type = W2VModel
    if args.fasttext:
        model_type = FastTextModel

    model_random = model_type.load(args.model_random)
    model_pretrained = model_type.load(args.model_pretrained)
    model_finetuned = model_type.load(args.model_finetuned)

    # print(f"Recall FULL random = {get_recall(train, test, Word2Vec(vector_size=300, min_count=1), args.topn)}")

    print(f"Recall random = {get_recall(train, test, model_random, args.topn)}")
    print(f"Recall pretrained = {get_recall(train, test, model_pretrained, args.topn)}")
    print(f"Recall finetuned = {get_recall(train, test, model_finetuned, args.topn)}")


if __name__ == "__main__":
    main(sys.argv[1:])
