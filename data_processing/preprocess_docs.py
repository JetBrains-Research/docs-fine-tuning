import argparse
import os.path

from pathlib import Path
from tika import parser
from util import remove_noise
from util import tokenize_and_normalize
from util import split_sentences


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--docs",
        dest="docs",
        action="extend",
        nargs="+",
        help="Paths to pdf docs to be preprocessed",
    )
    arg_parser.add_argument("-p", dest="prefix", action="store", help="Preprocessed docs file name prefix")
    return arg_parser.parse_args()


def get_text_from_pdf(file_name):
    raw = parser.from_file(file_name)
    return raw["content"]


def main():
    args = parse_arguments()
    for i, doc in enumerate(args.docs):
        text = get_text_from_pdf(doc)
        sentences = split_sentences(text)
        text = [remove_noise(sentence) for sentence in sentences]
        text = tokenize_and_normalize(text)
        with Path(os.path.join("data", "docs", f"{args.prefix}_{i}.txt")).open(mode="w") as f:
            f.write(str(text))


if __name__ == "__main__":
    main()
