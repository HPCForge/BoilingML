# BubbleML

[![Paper](https://img.shields.io/badge/arXiv-2209.15616-blue)](https://arxiv.org/abs/2307.14623)

A multiphysics, multiphase dataset of boiling processes.  
These models could be applied to model datacenter cooling systems. For instance, liquid cooling flowing across a GPU. or even cooling nuclear reactors: a pool of liquid sitting on a heated surface.

![SubCooled Temperature](video/subcooled.gif)

## Documentation and Examples

Documentation discussing the data fields, format, and relevant parameters can be found in [bubbleml_data/DOCS.md](bubbleml_data/DOCS.md). We also provide a set of [examples](examples/) illustrating how to use the dataset.


## Download BubbleML

BubbleML is publicly available and open source. We provide links to download each study in [bubbleml_data/README.md](bubbleml_data/README.md).

## Models

Checkpoints for all of the benchmark models mentioned in the paper, along with ther respective results are listed in the [model zoo](model-zoo/README.md)

## Running SciML Code

### Environment Setup
The code assumes access to a fairly modern Nvidia GPU, though
it may also work on AMD GPUs if PyTorch is installed with Rocm support.
Results have been reproduced on a Linux cluster with V100, A30, and A100 GPUs using PyTorch 2.0 and CUDA 11.7.

To install dependencies, we recommend creating a conda environment:

```console
conda env create -n bubble-sciml -f conda/pytorch-2.0.1-cuda-11.7.yaml
```

### Examples

In [examples/](examples/), we provide Jupyter Notebooks showing how to [read and visualize BubbleML](examples/data_loading.ipynb)
and [train a Fourier Neural Operator](examples/pytorch_training.ipynb). These are stand-alone examples that use a downsampled version of
Subcooled Pool boiling. These examples are intended to show 1. how to load the dataset, 2. how to read tensors from
the dataset, and 3. how to setup model training for the dataset. Extended descriptions can be found in [bubbleml_data/DOCS.md](bubbleml_data/DOCS.md)

### Experiment Code

Our sample application code uses Hydra to manage different configurations.
For example, we treat each simulation type as a dataset: `conf/dataset/*.yaml`.
Similarly, each model is treated as a separate experiment: `conf/experiment/*.yaml`.

For example, training a temperature prediction UNet model on the subcooled boiling dataset is simple:

```console
python src/train.py dataset=PB_SubCooled experiment=temp_unet2d
```

If you want to run a pretrained model, you can specify the `model_checkpoint` path

```console
python src/train.py dataset=PB_SubCooled experiment=temp_unet2d model_checkpoint=<path>
```

The config file `conf/default.yaml` assumes that the datasets are extracted to the same location.
**This location should be set by the user. By default, this setting is empty**.
Setting the `data_base_dir`  can be done by explicity updating `conf/default.yaml` or
specifying the dataset base directory when running the python scripts.) 

For example, if you downloaded two datasets to 

```console
/your/path/to/BubbleML/saturated.hdf5
/your/path/to/BubbleML/subcooled.hdf5
```

then, to train a UNet model on the subcooled boiling dataset, just run

```console
python src/train.py \
	data_base_dir=/your/path/to/BubbleML \
	dataset=PB_SubCooled experiment=temp_unet
```
## Running Optical Flow Benchmarks

Please refer to the [Optical Flow README.md](optical_flow/README.md)

## Citation

If you find this dataset useful in your research, please consider citing the following paper:

```bibtex
@article{hassan2023bubbleml,
      title={BubbleML: A Multi-Physics Dataset and Benchmarks for Machine Learning}, 
      author={Sheikh Md Shakeel Hassan and Arthur Feeney and Akash Dhruv and Jihoon Kim and 
	      Youngjoon Suh and Jaiyoung Ryu and Yoonjin Won and Aparna Chandramowlishwaran},
      year={2023},
      eprint={2307.14623},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
