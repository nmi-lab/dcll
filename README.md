# DCLL

## Install

It is recommended that you install python packages needed for your project in a python virtualenv.

To install **dcll**:
* activate your virtualenv;
* Clone the repo;
* Perform the install.

```bash
pip install -e .
```

By using **-e** option of pip, the files will be symlink'ed to your virtualenv instead of copied.
This means that you can modify the files of this repo without having to install it again for the changes to take effect.

## How to use

Check out some samples like `samples/pytorch_conv3L_mnist.py`.

### Tensorboard integration

We provide some utility functions for plotting to tensorboard.

```python
from tensorboardX import SummaryWriter
from dcll.pytorch_utils import NetworkDumper
import torch

net = torch.nn.Linear(100, 10)

ndumper = NetworkDumper(writer, net)

###### Plot activation from forward function
handle = ndumper.start_recording(title='activation')

for i in range(50):
    net.forward(torch.rand(100))

handle.remove()

###### Plot weight histogram
ndumper.histogram()

```
