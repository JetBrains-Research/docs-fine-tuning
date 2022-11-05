import argparse
import logging
import os.path
import tempfile
import json

from docs_preprocessor import DocsPreprocessor
from util import load_config


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--docs",
        dest="docs",
        action="extend",
        nargs="+",
        help="Paths to docs to be preprocessed",
    )
    arg_parser.add_argument(
        "--sfx", dest="suffix", action="store", default="prcsd", help="Preprocessed docs file name prefix"
    )
    return arg_parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()

    if config.tmpdir is not None:
        os.environ["TMPDIR"] = config.tmpdir
        tempfile.tempdir = config.tmpdir

    logging.basicConfig(
        filename=config.log_file, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    for docs_path in args.docs:
        preprocessor = DocsPreprocessor(docs_path, config.docs_formats)
        tokenized = preprocessor.preprocess_files()
        json_data = json.dumps(tokenized)

        result_file_name = os.path.join(
            config.docs_directory, os.path.splitext(os.path.split(docs_path)[1])[0] + "-" + args.suffix + ".json"
        )
        with open(result_file_name, "w") as doc:
            json.dump(json_data, doc)

        logger.info(f"Text artifacts from {docs_path} processed successfully and saved into {result_file_name}")


if __name__ == "__main__":
    main()
