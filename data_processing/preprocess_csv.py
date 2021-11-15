import sys
import argparse
import pandas as pd

from util import remove_noise
from util import tokenize_and_normalize


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        dest="full",
        action="store",
        help="The path to the full dataset to be preprocesse",
    )
    parser.add_argument(
        "--train",
        dest="train",
        action="store",
        help="The path to the file where the user wants " "to save the train portion of preprocessed " "dataset",
    )
    parser.add_argument(
        "--test",
        dest="test",
        action="store",
        help="The path to the file where the user wants " "to save the test portion of preprocessed " "dataset",
    )
    parser.add_argument(
        "--test_size",
        dest="test_size",
        action="store",
        type=float,
        default=0.2,
        help="The share of the test sample relative to the entire dataset",
    )
    return parser.parse_args(arguments)


def main(args_str):
    args = parse_arguments(args_str)
    data = pd.read_csv(args.full)

    data = data.dropna(axis=0, subset=["description"])
    data = data.reset_index(drop=True)
    data["description"] = data["description"].apply(remove_noise)
    data["description"] = data["description"].apply(tokenize_and_normalize)
    data = data.dropna(axis=0, subset=["description"])
    data = data.reset_index(drop=True)

    data_size = len(data.index)
    train_size = int((1 - args.test_size) * data_size)

    test = data.iloc[train_size:]
    test = test.reset_index(drop=True)
    test.to_csv(args.test, index=False)

    train = data.iloc[:train_size]
    train = train.reset_index(drop=True)
    train.to_csv(args.train, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
