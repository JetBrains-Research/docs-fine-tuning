import argparse
import os.path

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
        "--sfx", dest="suffix", action="store", default="-prcsd", help="Preprocessed docs file name prefix"
    )
    return arg_parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config()

    for docs_path in args.docs:
        preprocessor = DocsPreprocessor(docs_path, config.docs_formats)
        tokenized = preprocessor.preprocess_files()

        result_file_name = os.path.join(
            config.docs_directory, os.path.splitext(os.path.split(docs_path)[1])[0] + args.suffix + ".txt"
        )
        with open(result_file_name, "w") as doc:
            doc.write(str(tokenized))

        print(f"Text artifacts from {docs_path} processed successfully and saved into {result_file_name}")


if __name__ == "__main__":
    main()
