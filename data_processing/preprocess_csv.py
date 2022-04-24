import pandas as pd
from omegaconf import OmegaConf

from util import remove_noise, split_sentences
from util import tokenize_and_normalize, CONFIG_PATH


def main():
    config = OmegaConf.load(CONFIG_PATH)
    data = pd.read_csv(config.datasets.full)

    data = data.dropna(axis=0, subset=["description"])
    data = data.reset_index(drop=True)
    data["description"] = data["description"].apply(remove_noise)
    data["description"] = data["description"].apply(split_sentences)
    data["description"] = data["description"].apply(tokenize_and_normalize)
    data = data.dropna(axis=0, subset=["description"])
    data = data.reset_index(drop=True)

    data_size = len(data.index)
    train_size = int((1 - config.test_size) * data_size)

    test = data.iloc[train_size:]
    test = test.reset_index(drop=True)
    test.to_csv(config.datasets.test, index=False)
    print(f"Test size = {len(test.index)}")

    train = data.iloc[:train_size]
    train = train.reset_index(drop=True)
    train.to_csv(config.datasets.train, index=False)
    print(f"Train size = {len(train.index)}")


if __name__ == "__main__":
    main()
