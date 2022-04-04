import os
import argparse
import pandas as pd

from models import W2VModel, FastTextModel, BertModelMLM, SBertModel, RandomEmbeddingModel
from data_processing.util import get_corpus, load_config
from approaches import SimpleApproach, TfIdfApproach, IntersectionApproach


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
    parser.add_argument(
        "-w",
        dest="w",
        action="store",
        type=float,
        default=1,
        help="The weight of tf-idf vectors score in evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    if args.tfidf:
        evaluator = TfIdfApproach(train, test, args.w)
    elif args.intersection:
        evaluator = IntersectionApproach(train, test, args.min_count)
    else:
        evaluator = SimpleApproach(train, test)

    if args.intersection:
        print(f"Success Rate 'intersection' = {evaluator.evaluate(IntersectionApproach.UtilModel(), args.topn)}")
        return
    if args.random:
        model = RandomEmbeddingModel(get_corpus(train), min_count=args.min_count, w2v=args.w2v)
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

    from_scratch_model_path = (
        args.model_from_scratch
        if args.model_from_scratch
        else os.path.join(config["models_directory"], model_type.name + config["models"]["from_scratch"])
    )

    pretrained_model_path = (
        args.model_pretrained
        if args.model_pretrained
        else os.path.join(config["models_directory"], model_type.name + config["models"]["pretrained"])
    )

    finetuned_model_path = (
        args.model_finetuned
        if args.model_finetuned
        else os.path.join(config["models_directory"], model_type.name + config["models"]["fine-tuned"])
    )

    model_trained_from_scratch = model_type.load(from_scratch_model_path)
    model_pretrained = model_type.load(pretrained_model_path)
    model_finetuned = model_type.load(finetuned_model_path)

    print(f"Success Rate 'from scratch' = {evaluator.evaluate(model_trained_from_scratch, args.topn)}")
    print(f"Success Rate 'pretrained' = {evaluator.evaluate(model_pretrained, args.topn)}")
    print(f"Success Rate 'fine-tuned' = {evaluator.evaluate(model_finetuned, args.topn)}")


if __name__ == "__main__":
    main()
