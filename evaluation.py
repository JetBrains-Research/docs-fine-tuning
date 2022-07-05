import argparse
import logging
import os
import tempfile
import warnings
import pandas as pd

from approaches import SimpleApproach, TfIdfApproach, IntersectionApproach, FinetuningTasksTest
from data_processing.util import get_corpus, load_config
from text_models import (
    W2VModel,
    FastTextModel,
    BertModelMLM,
    SBertModel,
    RandomEmbeddingModel,
    BertSiameseModel,
    TrainTypes,
)

warnings.simplefilter(action="ignore", category=FutureWarning)  # for pd.DataFrame.append() method


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-id",
        dest="gpu_id",
        action="store",
        type=str,
        default="0",
        help="GPU id for CUDA_VISIBLE_DEVICES environment param",
    )
    return parser.parse_args()


def main():
    config = load_config()
    args = parse_arguments()
    cnf_eval = config.evaluation

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if config.tmpdir is not None:
        os.environ["TMPDIR"] = config.tmpdir
        tempfile.tempdir = config.tmpdir

    logging.basicConfig(
        filename=config.log_file, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

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

    if cnf_eval.is_tasks_test and cnf_eval.text_model == "siamese":
        evaluator = FinetuningTasksTest(
            evaluator,
            config.models.siamese.finetuning_strategies,
            config.models_directory,
            config.models_suffixes.finetuned,
        )

    if cnf_eval.approach == "intersection":
        logger.info(
            f"Success Rate 'intersection' = {evaluator.evaluate(IntersectionApproach.UtilModel(), cnf_eval.topns)}"
        )
        return

    if cnf_eval.text_model == "random":
        model = RandomEmbeddingModel(get_corpus(train), **config.models.random)
        logger.info(f"Success Rate 'random' = {evaluator.evaluate(model, cnf_eval.topns)}")
        return

    if cnf_eval.text_model == "word2vec":
        model_type = W2VModel
    elif cnf_eval.text_model == "fasttext":
        model_type = FastTextModel
    elif cnf_eval.text_model == "bert":
        model_type = BertModelMLM
    elif cnf_eval.text_model == "sbert":
        model_type = SBertModel
    elif cnf_eval.text_model == "siamese":
        model_type = BertSiameseModel
    else:
        raise ValueError(f"Text model ${cnf_eval.text_model} is not supported")

    model_trained_from_scratch = None
    model_pretrained = None
    model_doc_task = None
    model_finetuned = None

    if TrainTypes.TASK in config.model_types:
        model_trained_from_scratch = model_type.load(
            os.path.join(config.models_directory, model_type.name + config.models_suffixes.from_scratch)
        )

    if TrainTypes.PT_TASK in config.model_types:
        model_pretrained = model_type.load(
            os.path.join(config.models_directory, model_type.name + config.models_suffixes.pretrained)
        )

    task_name = "" if cnf_eval.text_model != "siamese" else "_" + config.models.siamese.finetuning_strategies[0]
    if TrainTypes.DOC_TASK in config.model_types:
        model_doc_task = model_type.load(
            os.path.join(config.models_directory, model_type.name + task_name + config.models_suffixes.doc_task)
        )

    if TrainTypes.PT_DOC_TASK in config.model_types:
        model_finetuned = model_type.load(
            os.path.join(config.models_directory, model_type.name + task_name + config.models_suffixes.finetuned)
        )

    evaluator.evaluate_all(
        model_trained_from_scratch, model_pretrained, model_doc_task, model_finetuned, cnf_eval.topns
    )
    if cnf_eval.save_results:
        evaluator.save_results(cnf_eval.results_path, model_type.name, plot=cnf_eval.save_graph)


if __name__ == "__main__":
    main()
