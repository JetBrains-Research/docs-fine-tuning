import argparse
import os.path
from pathlib import Path

from omegaconf import OmegaConf
from tika import parser

from util import CONFIG_PATH
from util import remove_noise, tokenize_and_normalize, split_sentences


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--docs",
        dest="docs",
        action="extend",
        nargs="+",
        help="Paths to pdf docs to be preprocessed",
    )
    arg_parser.add_argument(
        "-p", dest="prefix", action="store", default="doc", help="Preprocessed docs file name prefix"
    )
    return arg_parser.parse_args()


def get_text_from_pdf(file_name):
    raw = parser.from_file(file_name)
    return raw["content"]


def main():
    args = parse_arguments()
    config = OmegaConf.load(CONFIG_PATH)
    for i, doc in enumerate(args.docs):
        text = get_text_from_pdf(doc)
        text = remove_noise(text)
        sentences = split_sentences(text)
        text = tokenize_and_normalize(sentences)
        with Path(os.path.join(config.docs_directory, f"{args.prefix}_{i}.txt")).open(mode="w") as f:
            f.write(str(text))


if __name__ == "__main__":
    main()
