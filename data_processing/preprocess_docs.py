import sys
import argparse
import os.path

from pathlib import Path
from tika import parser
from util import remove_noise
from util import tokenize_and_normalize


def parse_arguments(arguments):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--docs",
        dest="docs",
        action="extend",
        nargs="+",
        help="Paths to pdf docs to be " "preprocessed",
    )
    return arg_parser.parse_args(arguments)


def get_text_from_pdf(file_name):
    raw = parser.from_file(file_name)
    return raw["content"]


def main(args_str):
    args = parse_arguments(args_str)
    for i, doc in enumerate(args.docs):
        text = get_text_from_pdf(doc)
        text = remove_noise(text)
        text = tokenize_and_normalize(text)
        with Path(os.path.join("data", "docs", f"doc_{i}.txt")).open(mode="w") as f:
            f.write(str(text))


if __name__ == "__main__":
    main(sys.argv[1:])
