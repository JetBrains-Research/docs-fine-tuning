import sys
import argparse
import os.path
import nltk.data

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
    arg_parser.add_argument("-p", dest="prefix", action="store", help="Preprocessed docs file name prefix")
    return arg_parser.parse_args(arguments)


def get_text_from_pdf(file_name):
    raw = parser.from_file(file_name)
    return raw["content"]


def main(args_str):
    args = parse_arguments(args_str)
    for i, doc in enumerate(args.docs):
        text = get_text_from_pdf(doc)
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        text = []
        for sentence in sentences:
            sentence = remove_noise(sentence)
            sentence = tokenize_and_normalize(sentence)
            text.append(sentence)
        with Path(os.path.join("data", "docs", f"{args.prefix}_{i}.txt")).open(mode="w") as f:
            f.write(str(text))


if __name__ == "__main__":
    main(sys.argv[1:])
