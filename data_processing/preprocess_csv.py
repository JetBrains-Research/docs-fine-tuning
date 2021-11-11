import sys
import argparse
import pandas as pd

from util import remove_noise
from util import tokenize_and_normalize


def parse_arguments(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', dest='full', action='store')
    parser.add_argument('--train', dest='train', action='store')
    parser.add_argument('--test', dest='test', action='store')
    parser.add_argument('--test_size', dest='test_size', action='store', type=float, default=0.2)
    return parser.parse_args(arguments)


def main(args_str):
    data = pd.read_csv(args_str[0])
    args = parse_arguments(args_str[1:])

    data = data.drop(
        columns=['Priority', 'Component', 'Status', 'Resolution', 'Version', 'Created_time', 'Resolved_time'], axis=1)
    data = data.dropna(axis=0, subset=['Description'])
    data['Title'] = data['Title'].apply(remove_noise)
    data['Description'] = data['Description'].apply(remove_noise)
    data['Title'] = data['Title'].apply(tokenize_and_normalize)
    data['Description'] = data['Description'].apply(tokenize_and_normalize)
    data = data.dropna(axis=0, subset=['Description'])

    data.to_csv(args.full)
    data_size = len(data.index)
    train_size = int((1 - args.test_size) * data_size)
    data.iloc[train_size:].to_csv(args.test)
    data.iloc[:train_size].to_csv(args.train)


if __name__ == '__main__':
    main(sys.argv[1:])
