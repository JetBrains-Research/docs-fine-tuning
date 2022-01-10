import argparse
import pandas as pd

from util import remove_noise, split_sentences
from util import tokenize_and_normalize


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        dest="full",
        action="store",
        help="The path to the full dataset to be preprocessed",
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    data = pd.read_csv(args.full)

    data = data.dropna(axis=0, subset=["description"])
    data = data.reset_index(drop=True)
    data["description"] = data["description"].apply(remove_noise)
    data["description"] = data["description"].apply(split_sentences)
    data["description"] = data["description"].apply(tokenize_and_normalize)
    data = data.dropna(axis=0, subset=["description"])
    data = data.reset_index(drop=True)

    data_size = len(data.index)
    train_size = int((1 - args.test_size) * data_size)

    test = data.iloc[train_size:]
    test = test.reset_index(drop=True)
    test.to_csv(args.test, index=False)
    print(f"Test size = {len(test.index)}")

    train = data.iloc[:train_size]
    train = train.reset_index(drop=True)
    train.to_csv(args.train, index=False)
    print(f"Train size = {len(train.index)}")


if __name__ == "__main__":
    main()
