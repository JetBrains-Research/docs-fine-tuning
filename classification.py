import argparse
import numpy as np
import pandas as pd

from models import AbstractModel, W2VModel, FastTextModel, BertModelMLM, SBertModel
from data_processing.util import get_corpus

import faiss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument("--test", dest="test", action="store", help="The path to test dataset")
    parser.add_argument("-r", dest="model_random", action="store", help="The path to random model")
    parser.add_argument(
        "-p", dest="model_pretrained", action="store", help="The path to pretrained model",
    )
    parser.add_argument(
        "-f", dest="model_finetuned", action="store", help="The path to fine-tuned model",
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
    parser.add_argument("--bert", dest="bert", action="store_true", help="Use BERT model for classification")
    parser.add_argument("--sbert", dest="sbert", action="store_true", help="Use SBERT model for classification")
    return parser.parse_args()


def get_recall(train: pd.DataFrame, test: pd.DataFrame, model: AbstractModel, topn: int):

    train_corpus = get_corpus(train)
    embeddings = model.get_embeddings(train_corpus).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])

    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    test_corpus = get_corpus(test)
    test_embs = model.get_embeddings(test_corpus, update_vocab=False).astype(np.float32)

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


def main():
    args = parse_arguments()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    if args.w2v:
        model_type = W2VModel
    elif args.fasttext:
        model_type = FastTextModel
    elif args.bert:
        model_type = BertModelMLM
    elif args.sbert:
        model_type = SBertModel
    else:
        raise ValueError("Please select a model")

    model_trained_from_scratch = model_type.load(args.model_random)
    model_pretrained = model_type.load(args.model_pretrained)
    model_finetuned = model_type.load(args.model_finetuned)

    print(f"Recall 'from scratch' = {get_recall(train, test, model_trained_from_scratch, args.topn)}")
    print(f"Recall 'pretrained' = {get_recall(train, test, model_pretrained, args.topn)}")
    print(f"Recall 'fine-tuned' = {get_recall(train, test, model_finetuned, args.topn)}")


if __name__ == "__main__":
    main()
