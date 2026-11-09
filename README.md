# PSIAL

This repository provides the PyTorch implementation of **PSIAL**. We hope that this code will be useful for your research and applications.

## Requirements

The code is implemented and tested with the following environment:

- Python 3.9.12
- PyTorch 1.11.0
- NumPy 1.21.2

## Datasets

We provide preprocessed datasets, which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1-5hKD4hKd0eRdagm4MBW1g5kjH5qgmHR?usp=sharing).

For the original datasets, please refer to the [STSGCN repository](https://github.com/Davidham3/STSGCN), which includes PEMS03, PEMS04, PEMS07, and PEMS08.

We sincerely thank the providers of these datasets for making them publicly available.

## Training

Before training, **run `kshape_cluster_traffic.py` separately in each dataset directory** to generate the corresponding K-Shape pattern libraries.

For example, run the following commands:

```bash
cd /data/PEMS08
python kshape_cluster_traffic.py

```

After generating the pattern libraries, you can start model training.

The following example shows how to train PSIAL on PEMS08:

```bash
nohup python -u train.py --data PEMS08 > PEMS08.log 2>&1 &
```

You can customize the model configuration and training hyperparameters in `train.py`.

## Reference Code

The implementation was developed based on the [LibCity](https://github.com/Aoyu-Liu/STAMT) framework, an open-source library for traffic prediction.
