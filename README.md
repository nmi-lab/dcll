# Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DCLL)

This repo contains the pytorch implementation of the DCLL learning rule presented in [this paper](https://arxiv.org/abs/1811.10766).
If you use this code in a scientific publication, please include the following reference in your bibliography:

```
@article{kaiser2018synaptic,
  title={Synaptic Plasticity Dynamics for Deep Continuous Local Learning},
  author={Kaiser, Jacques and Mostafa, Hesham and Neftci, Emre},
  journal={arXiv preprint arXiv:1811.10766},
  year={2018}
}
```

## Install

This repo is a python package.
You can install it in a virtual environment (or locally with `--user`) with the following command:

```bash
pip install -e .
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
ln -s ~/Datasets/DVS/dvs_gestures/DVS\ \ Gesture\ dataset/DvsGesture/ .
```

## Code

The core of the DCLL framework are implemented as torch Modules in `dcll/pytorch_libdcll.py`.
