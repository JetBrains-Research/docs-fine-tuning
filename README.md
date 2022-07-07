# Fine-Tuning Text Models on Docs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI](https://github.com/Readrid/aug-text-to-sql/actions/workflows/ubuntu-python.yml/badge.svg)


This repository contains a machine learning pipeline for solving the problem of finding duplicate bug reports 
using text models trained on the documentation of the corresponding projects. 

You can see available text models and solution approaches in `config.yml`.

## Data Preparation

To use this pipeline, your data must look like this 

| id | ... |  description  | ... | disc_id |
|----|-----|:-------------:|-----|:-------:|
|... | ... |      ...      | ... |   ...   |
| 33 | ... | this is a bug | ... |   11    |

where **disc_id** is the id of the oldest duplicate bug report.

You can configure your dataset paths with `data/datasets_config.yml` and select them in `config.yml`.
To preprocess your csv data, use this command:

```shell
$ python data_processing/preprocess_csv.py
```

To preprocess docs, use this command:
```shell
$ python data_processing/preprocess_docs.py --docs <doc_path> [<doc_path> ...] 
```

## Training

There are 4 training approaches available:

* **TASK**: train from scratch on the task of finding duplicate bug reports
* **PT+TASK**: train **TASK** using a pre-trained text model
* **DOC+TASK**: train from scratch on docs and then train on the task of finding duplicate bug reports
* **PT+DOC+TASK**: train **DOC+TASK** using a pre-trained text model

You can configure training approaches and models with `config.yml`.

To run training, use this command:

```shell
$ python train_models.py <text-model-type> --docs <processed-doc-path> [<processed-doc-path> ...]
```

## Evaluation

To calculate quality metrics, use this command:

```shell
$ python evaluation.py
```

You can configure the results' path with `config.yml`. 