import sys
import argparse
import numpy as np
import pandas as pd

from gensim.models.word2vec import Word2Vec
from data_processing.util import get_corpus


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store", help="The path to train dataset")
    parser.add_argument("--test", dest="test", action="store", help="The path to test dataset")
    parser.add_argument("-r", dest="model_random", action="store", help="The path to random model")
    parser.add_argument("-p", dest="model_pretrained", action="store", help="The path to pretrained model")
    parser.add_argument("-f", dest="model_finetuned", action="store", help="The path to fine-tuned model")
    parser.add_argument("--topn", dest="topn", action="store", type=int, default=5,
                        help="The number of predicted duplicate bug-reports for one report")
    return parser.parse_args(arguments)


def get_doc_embedding(doc, model):
    result = np.zeros(model.vector_size)
    for word in doc:
        result += model[word]

    return result / len(doc)


def get_reports_embeddings(model, data):
    embeddings = []
    corpus = get_corpus(data)
    for report in corpus:
        embeddings.append(get_doc_embedding(report, model))
    return embeddings


def sim(vec1, vec2):
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_top_duplicate(bug_descr, embeddings, model, topn):
    doc_emb = get_doc_embedding(bug_descr, model)
    sims = []
    for report_emb in embeddings:
        sims.append(sim(report_emb, doc_emb))
    sims = np.array(sims)
    sims = sims.argsort()[::-1][:topn]
    return sims


def get_recall(data, test, model, master_ids, topn):
    embeddings = get_reports_embeddings(model.wv, data)
    test_size = 0
    test_corpus = get_corpus(test)
    TP = 0.
    model.build_vocab(test_corpus, update=True)
    for ind, descr in enumerate(test_corpus):
        if test.iloc[ind]['id'] not in master_ids:
            dupl_ids = find_top_duplicate(descr, embeddings, model.wv, topn)
            val = 0.
            for dupl_id in dupl_ids:
                if data.iloc[dupl_id]['disc_id'] == test.iloc[ind]['disc_id']:
                    val = 1.
                    break
            TP += val
            test_size += 1

        data = data.append(test.iloc[ind], ignore_index=True)
        embeddings.append(get_doc_embedding(descr, model.wv))
    return TP / test_size


def main(args_str):
    args = parse_arguments(args_str)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    master_ids_from_test = np.unique(test["disc_id"].to_numpy())

    model_random = Word2Vec.load(args.model_random)
    model_pretrained = Word2Vec.load(args.model_pretrained)
    model_finetuned = Word2Vec.load(args.model_finetuned)

    print(f"Recall random = {get_recall(train, test, model_random, master_ids_from_test, args.topn)}")
    print(f"Recall pretrained = {get_recall(train, test, model_pretrained, master_ids_from_test, args.topn)}")
    print(f"Recall finetuned = {get_recall(train, test, model_finetuned, master_ids_from_test, args.topn)}")


if __name__ == "__main__":
    main(sys.argv[1:])
