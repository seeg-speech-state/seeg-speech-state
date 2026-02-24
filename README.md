# Robust Decoding of Human Speech State via Cortical and Subcortical SEEG Sparse Signals
[![license](https://img.shields.io/badge/license-Apache--2.0-%23B7A800)](LICENSE)

## Introduction

The main branch works with **PyTorch 2.5.1** or higher.

### What does this repo do?

This is the official implementation of the paper **Robust Decoding of Human Speech State via Cortical and Subcortical SEEG Sparse Signals**.


## Installation

There are quick installation steps for develepment:

```shell
conda create -n seeg python=3.12 -y
conda activate seeg
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 # as an example
```

Installation from scratch typically takes less than 2 hours.

## Training

We present two Python scripts for training and evaluation. Run the training script with help command to check the available parameters and their default values:
```shell
python main.py --help
```

To evaluate the trained models, run the following script:
```shell
python infer.py --help
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
