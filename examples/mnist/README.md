# MNIST Classification

Build and train simple neural network from MNIST image dataset.
The train script looks for [mnist-dataset files](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) under directory `./_local/`.

The files can be downloaded with the Kaggle CLI with following command:

```sh
mkdir -p _local
kaggle datasets download hojjatk/mnist-dataset -p _local --unzip
```

## How to run

```sh
deno run train.ts
```
