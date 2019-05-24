# Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)

This repo contains the [PyTorch](https://pytorch.org/) implementation of the DECOLLE learning rule presented in [this paper](https://arxiv.org/abs/1811.10766).
If you use this code in a scientific publication, please include the following reference in your bibliography:

```
@article{kaiser2018synaptic,
  title={Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE)},
  author={Kaiser, Jacques and Mostafa, Hesham and Neftci, Emre},
  journal={arXiv preprint arXiv:1811.10766},
  year={2018}
}
```

## Install

This repo is a python package depending on [PyTorch](https://pytorch.org/).
You can install it in a virtual environment (or locally with `--user`) with the following command:

```bash
pip install -e .
pip install -r requirements.txt
```

By using the `-e` option of pip, the files will be symlink'ed to your virtualenv instead of copied.
This means that you can modify the files of this repo without having to install it again for the changes to take effect.

## Run

You can reproduce the results presented in the [original paper](https://arxiv.org/abs/1811.10766) by running the following scripts:

* `samples/pytorch_conv3L_mnist.py` (or `samples/run_mnist_experiments.sh`)
* `samples/pytorch_conv3L_dvsgestures_args.py`

The latest requires you to [download the DVS gesture dataset](http://research.ibm.com/dvsgesture/) and move the `DvsGesture` folder into the `data` folder:

```
cd data
ln -s path_to_dvs_gestures/DVS\ \ Gesture\ dataset/DvsGesture/ .
```
## Browse the results

You can browse our results without having to run the code.
Check out the jupyter notebooks for the [DvsGesture dataset](notebooks/plot_dvs_gestures.ipynb) and the [MNIST dataset](notebooks/plot_mnist.ipynb).

## Usage

The core of the DECOLLE framework are implemented as torch Modules in `dcll/pytorch_libdcll.py`.
Since we provide this code as a library, you can reuse the DECOLLE layers in your own [PyTorch](https://pytorch.org/) code:

```
import torch
from dcll.pytorch_libdcll import Conv2dDCLLlayer
layer = Conv2dDCLLlayer(...)
```

Have a look in [samples/pytorch_conv3L_mnist.py](samples/pytorch_conv3L_mnist.py) for example usage.

## Tutorials

The notebooks under tutorials are standalone, step-by-step instructions for setting up spiking neural networks in PyTorch and setting up DECOLLE. See for example [tutorials/dcll_tutorial1.ipynb](tutorials/dcll_tutorial1.ipynb).
