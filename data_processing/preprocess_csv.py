import logging
import os
import tempfile

import pandas as pd

from util import preprocess, load_config


def main():
    config = load_config()
    if config.tmpdir is not None:
        os.environ["TMPDIR"] = config.tmpdir
        tempfile.tempdir = config.tmpdir

    logging.basicConfig(
        filename=config.log_file, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    data = pd.read_csv(config.datasets.full)

    data["summary"] = data["summary"].map(preprocess, na_action="ignore")
    data["description"] = data["description"].map(preprocess, na_action="ignore")

    data = data[~(data.summary.isnull() & data.description.isnull())]
    data = data.reset_index(drop=True)

    data_size = len(data.index)
    train_size = int((1 - config.test_size) * data_size)

    test = data.iloc[train_size:]
    test = test.reset_index(drop=True)
    test.to_csv(config.datasets.test, index=False)
    logger.info(f"Test size = {len(test.index)}")

    train = data.iloc[:train_size]
    train = train.reset_index(drop=True)
    train.to_csv(config.datasets.train, index=False)
    logger.info(f"Train size = {len(train.index)}")


if __name__ == "__main__":
    main()
