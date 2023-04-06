# Fine-Tuning Text Models on Docs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![CI](https://github.com/JetBrains-Research/docs-fine-tuning/actions/workflows/ubuntu-python.yml/badge.svg)


This repository contains a machine learning pipeline for solving the problem of finding duplicate bug reports 
using text models trained on the documentation of the corresponding projects. 

You can see available text models and solution approaches in [`config.yml`](config.yml).

## Data Preparation

To use this pipeline, your data should be in csv format and must look like this 

|  id  | ... |   summary   |  ...  |  description  |  ...  | disc_id |
|:----:|-----|:-----------:|:-----:|:-------------:|:-----:|:-------:|
| ...  | ... |     ...     |  ...  |      ...      |  ...  |   ...   |
|  33  | ... | bug example |  ...  | this is a bug |  ...  |   11    |

where **disc_id** is the id of the oldest duplicate bug report.

You can configure your dataset paths with [`data/datasets_config.yml`](data/datasets_config.yml) and select them in [`config.yml`](config.yml).
To preprocess your csv data, use this command:

```shell
$ python data_processing/preprocess_csv.py
```

To preprocess docs, use this command:
```shell
$ python data_processing/preprocess_docs.py --docs <doc_path> [<doc_path> ...] 
```

## Training

There are 8 training approaches available:

* **TASK**: train from scratch on the task of finding duplicate bug reports
* **PT+TASK**: train **TASK** using a pre-trained text model
* **DOC+TASK**: train from scratch on docs and then train on the task of finding duplicate bug reports
* **BUGS+TASK**: train from scratch on bugs descriptions and then train on the task of finding duplicate bug reports
* **DOC+BUGS+TASK**: train from scratch on docs and bugs descriptions and then train on the task of finding duplicate bug reports
* **PT+DOC+TASK**: train **DOC+TASK** using a pre-trained text model
* **PT+BUGS+TASK**: train **BUGS+TASK** using a pre-trained text model
* **PT+DOC+BUGS+TASK**: train **DOC+BUGS+TASK** using a pre-trained text model

You can configure training approaches and models with [`config.yml`](config.yml).

To run training, use this command:

```shell
$ python train_models.py --gpu-id <id>
```

## Evaluation

To calculate quality metrics, use this command:

```shell
$ python evaluation.py --gpu-id <id>
```

You can configure the results' path with [`config.yml`](config.yml).

## Tune Hyperparameters

To tune the hyperparameters of the siamese model for a specific TASK, you can use [wandb sweeps](https://docs.wandb.ai/guides/sweeps). Follow the steps below:

1. Configure the model to tune and its parameters with the [`sweep_config.yaml`](sweep_config.yaml) file.
2. Run the following command to start the sweep
   ```shell
   $ wandb sweep sweep_config.yaml
   ```
3. You will see a <sweep_id> in the output. Copy this ID and run the following command:
   ```shell
   $ CUDA_VISIBLE_DEVICES=<gpu_id> wandb agent --count <runs_number> <sweep_id>
   ```
