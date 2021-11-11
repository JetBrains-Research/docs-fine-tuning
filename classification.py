import sys
import argparse
import numpy as np
import pandas as pd

from gensim.models.word2vec import Word2Vec
from data_processing.util import get_corpus


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action='store')
    parser.add_argument("--test", dest="test", action="store")
    parser.add_argument("-r", dest="model_random", action="store")
    parser.add_argument("-p", dest="model_pretrained", action="store")
    parser.add_argument("-f", dest="model_finetuned", action="store")
    parser.add_argument("--topn", dest="topn", action="store", type=int, default=5)
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
    return np.linalg.norm(vec1 - vec2)


def find_top_duplicate(bug_descr, embeddings, model, topn):
    doc_emb = get_doc_embedding(bug_descr, model)
    sims = []
    for report_emb in embeddings:
        sims.append(sim(report_emb, doc_emb))
    sims = np.array(sims)
    sims = sims.argsort()[:topn]
    return sims


def get_recall(train, test, model, topn):
    embeddings = get_reports_embeddings(model.wv, train)
    test_size = len(test.index)
    test_corpus = get_corpus(test)
    model.build_vocab(test_corpus, update=True)
    TP = 0.
    for ind, descr in enumerate(test_corpus):
        dupl_ids = find_top_duplicate(descr, embeddings, model.wv, topn=topn)
        val = 0.
        for dupl_id in dupl_ids:
            if train.iloc[dupl_id]['Issue_id'] == test.iloc[ind]['Duplicated_issue']:
                val = 1.
                break
        TP += val
    return TP / test_size


def main(args_str):
    args = parse_arguments(args_str)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    model_random = Word2Vec.load(args.model_random)
    model_pretrained = Word2Vec.load(args.model_pretrained)
    model_finetuned = Word2Vec.load(args.model_finetuned)

    master_reports = train[train.isnull().any(axis=1)]
    dupl_test_reports = test.dropna(axis=0, subset=['Duplicated_issue'])

    print(f"Recall random = {get_recall(master_reports, dupl_test_reports, model_random, args.topn)}")
    print(f"Recall pretrained = {get_recall(master_reports, dupl_test_reports, model_pretrained, args.topn)}")
    print(f"Recall finetuned = {get_recall(master_reports, dupl_test_reports, model_finetuned, args.topn)}")


if __name__ == '__main__':
    main(sys.argv[1:])


