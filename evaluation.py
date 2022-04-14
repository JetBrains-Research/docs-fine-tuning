import os

import pandas as pd

from approaches import SimpleApproach, TfIdfApproach, IntersectionApproach
from data_processing.util import get_corpus, load_config
from text_models import W2VModel, FastTextModel, BertModelMLM, SBertModel, RandomEmbeddingModel


def main():
    config = load_config()
    cnf_eval = config.evaluation

    train = pd.read_csv(config.datasets.train)
    test = pd.read_csv(config.datasets.test)

    if cnf_eval.approach == "tf_idf":
        evaluator = TfIdfApproach(train, test, config.approaches.tf_idf.weight)
    elif cnf_eval.approach == "intersection":
        evaluator = IntersectionApproach(train, test, config.approaches.intersection.min_count)
    elif cnf_eval.approach == "simple":
        evaluator = SimpleApproach(train, test)
    else:
        raise ValueError(f"Approach ${cnf_eval.approach} is not supported")

    if cnf_eval.approach == "intersection":
        print(f"Success Rate 'intersection' = {evaluator.evaluate(IntersectionApproach.UtilModel(), config.topns)}")
        return

    if cnf_eval.text_model == "random":
        model = RandomEmbeddingModel(get_corpus(train), **config.models.random)
        print(f"Success Rate 'random' = {evaluator.evaluate(model, config.topns)}")
        return

    if cnf_eval.text_model == "word2vec":
        model_type = W2VModel
    elif cnf_eval.text_model == "fasttext":
        model_type = FastTextModel
    elif cnf_eval.text_model == "bert":
        model_type = BertModelMLM
    elif cnf_eval.text_model == "sbert":
        model_type = SBertModel
    else:
        raise ValueError(f"Text model ${cnf_eval.text_model} is not supported")

    model_trained_from_scratch = model_type.load(
        os.path.join(config.models_directory, model_type.name + config.models_suffixes.from_scratch)
    )
    model_pretrained = model_type.load(
        os.path.join(config.models_directory, model_type.name + config.models_suffixes.pretrained)
    )
    model_finetuned = model_type.load(
        os.path.join(config.models_directory, model_type.name + config.models_suffixes.finetuned)
    )

    evaluator.evaluate_all(model_trained_from_scratch, model_pretrained, model_finetuned, config.topns)
    if config.save_results:
        evaluator.save_results(config.results_path, model_type.name, graph=config.save_graph)


if __name__ == "__main__":
    main()
