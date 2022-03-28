import argparse
import numpy as np
import pandas as pd
from nltk import FreqDist

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack

from models import AbstractModel, W2VModel, FastTextModel, BertModelMLM, SBertModel
from models import RandomEmbeddingModel
from data_processing.util import get_corpus

import faiss


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument("--test", dest="test", action="store", help="The path to test dataset")
    parser.add_argument("-s", dest="model_from_scratch", action="store", help="The path to model training from scratch")
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
    parser.add_argument("--bert", dest="bert", action="store_true", help="Use BERT model for classification")
    parser.add_argument("--sbert", dest="sbert", action="store_true", help="Use SBERT model for classification")
    parser.add_argument("--random", dest="random", action="store_true", help="Use random embeddings for classification")
    parser.add_argument(
        "--intersection",
        dest="intersection",
        action="store_true",
        help="Use word intersections for success rate evaluation",
    )
    parser.add_argument(
        "--min_count",
        dest="min_count",
        action="store",
        type=int,
        default=1,
        help="Ignore all words with total frequency lower than this in intersection mode",
    )
    parser.add_argument("--tfidf", dest="tfidf", action="store_true", help="Use tf-idf matrix in evaluation")
    return parser.parse_args()


def get_success_rate_by_intersection(train: pd.DataFrame, test: pd.DataFrame, min_count: int, topn: int):
    train_corpus = get_corpus(train)
    test_corpus = get_corpus(test)

    freq_dict = FreqDist()
    for report in train_corpus:
        freq_dict.update(report)
    for report in test_corpus:
        freq_dict.update(report)
    test_corpus = [list(filter(lambda x: freq_dict[x] >= min_count, report)) for report in test_corpus]
    train_corpus = [list(filter(lambda x: freq_dict[x] >= min_count, report)) for report in train_corpus]

    test_size = 0
    TP = 0
    for ind, descr in enumerate(test_corpus):
        if test.iloc[ind]["id"] != test.iloc[ind]["disc_id"]:

            counts = []
            for report in train_corpus:
                count = len(list(set(report) & set(descr)))
                counts.append(count)
            dupl_ids = np.argsort(counts)[::-1][:topn]

            val = 0
            for dupl_id in dupl_ids:
                if train.iloc[dupl_id]["disc_id"] == test.iloc[ind]["disc_id"]:
                    val = 1
                    break
            TP += val
            test_size += 1

        train_corpus.append(test_corpus[ind])
        train = train.append(test.iloc[ind], ignore_index=True)
    return TP / test_size


def get_success_rate_with_tfidf(train: pd.DataFrame, test: pd.DataFrame, model: AbstractModel, topn: int, w: int = 0.7):
    train_corpus = get_corpus(train)
    embeddings = model.get_embeddings(train_corpus)

    test_corpus = get_corpus(test)
    test_embs = model.get_embeddings(test_corpus)

    tf_idf = TfidfVectorizer(use_idf=True)
    train_corpus_tf_idf = [" ".join(doc) for doc in train_corpus]
    tf_idf.fit(train_corpus_tf_idf)

    train_tfidf_vectors = tf_idf.transform(train_corpus_tf_idf)
    test_tfidf_vectors = tf_idf.transform([" ".join(doc) for doc in test_corpus])

    test_size = 0
    TP = 0

    for ind, descr in enumerate(test_corpus):
        if test.iloc[ind]["id"] != test.iloc[ind]["disc_id"]:  # not in master_ids
            sims_emb = cosine_similarity(embeddings, test_embs[ind].reshape(1, -1)).squeeze()
            sims_tfidf = cosine_similarity(train_tfidf_vectors, test_tfidf_vectors[ind]).squeeze()
            sims = w * sims_tfidf + (1 - w) * sims_emb
            dupl_ids = np.argsort(-sims)[:topn]
            val = 0
            for dupl_id in dupl_ids:
                if train.iloc[dupl_id]["disc_id"] == test.iloc[ind]["disc_id"]:
                    val = 1
                    break
            TP += val
            test_size += 1

        train = train.append(test.iloc[ind], ignore_index=True)
        train_tfidf_vectors = vstack((train_tfidf_vectors, test_tfidf_vectors[ind]))
        embeddings = np.append(embeddings, test_embs[ind].reshape(1, -1), axis=0)

    return TP / test_size


def get_success_rate(train: pd.DataFrame, test: pd.DataFrame, model: AbstractModel, topn: int):
    train_corpus = get_corpus(train)
    embeddings = model.get_embeddings(train_corpus)
    index = faiss.IndexFlatIP(embeddings.shape[1])

    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    test_corpus = get_corpus(test)
    test_embs = model.get_embeddings(test_corpus)

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

    evaluate = get_success_rate_with_tfidf if args.tfidf else get_success_rate

    if args.intersection:
        print(
            f"Success Rate 'intersection' = {get_success_rate_by_intersection(train, test, args.min_count, args.topn)}"
        )
        return
    if args.random:
        model = RandomEmbeddingModel(get_corpus(train), min_count=args.min_count, w2v=args.w2v)
        print(f"Success Rate 'random' = {evaluate(train, test, model, args.topn)}")
        return
    elif args.w2v:
        model_type = W2VModel
    elif args.fasttext:
        model_type = FastTextModel
    elif args.bert:
        model_type = BertModelMLM
    elif args.sbert:
        model_type = SBertModel
    else:
        raise ValueError("Please select a model")

    model_trained_from_scratch = model_type.load(args.model_from_scratch)
    model_pretrained = model_type.load(args.model_pretrained)
    model_finetuned = model_type.load(args.model_finetuned)

    print(f"Success Rate 'from scratch' = {evaluate(train, test, model_trained_from_scratch, args.topn)}")
    print(f"Success Rate 'pretrained' = {evaluate(train, test, model_pretrained, args.topn)}")
    print(f"Success Rate 'fine-tuned' = {evaluate(train, test, model_finetuned, args.topn)}")


if __name__ == "__main__":
    main()
