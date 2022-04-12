import argparse
import os

import pandas as pd

from approaches import SimpleApproach, TfIdfApproach, IntersectionApproach
from data_processing.util import get_corpus, load_config
from text_models import W2VModel, FastTextModel, BertModelMLM, SBertModel, RandomEmbeddingModel


def parse_arguments():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--tfidf", dest="tfidf", action="store_true", help="Use tf-idf matrix in evaluation")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()

    train = pd.read_csv(config.datasets.train)
    test = pd.read_csv(config.datasets.test)

    if args.tfidf:
        evaluator = TfIdfApproach(train, test, config.approaches.tf_idf.weight)
    elif args.intersection:
        evaluator = IntersectionApproach(train, test, config.approaches.intersection.min_count)
    else:
        evaluator = SimpleApproach(train, test)

    if args.intersection:
        print(f"Success Rate 'intersection' = {evaluator.evaluate(IntersectionApproach.UtilModel(), args.topn)}")
        return
    if args.random:
        cnf_random = config.models.random
        model = RandomEmbeddingModel(get_corpus(train), min_count=cnf_random.min_count, w2v=cnf_random.rand_by_w2v)
        print(f"Success Rate 'random' = {evaluator.evaluate(model, args.topn)}")
        return
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

    model_trained_from_scratch = model_type.load(
        os.path.join(config.models_directory, model_type.name + config.models_suffixes.from_scratch)
    )
    model_pretrained = model_type.load(
        os.path.join(config.models_directory, model_type.name + config.models_suffixes.pretrained)
    )
    model_finetuned = model_type.load(
        os.path.join(config.models_directory, model_type.name + config.models_suffixes.finetuned)
    )

    print(f"Success Rate 'from scratch' = {evaluator.evaluate(model_trained_from_scratch, args.topn)}")
    print(f"Success Rate 'pretrained' = {evaluator.evaluate(model_pretrained, args.topn)}")
    print(f"Success Rate 'fine-tuned' = {evaluator.evaluate(model_finetuned, args.topn)}")


if __name__ == "__main__":
    main()
