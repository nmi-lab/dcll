#!/bin/python
#-----------------------------------------------------------------------------
# File Name : pytorch_dense1L_mnist.py
# Author: Emre Neftci
#
# Creation Date : Wed 18 Jul 2018 02:39:06 PM MDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from pytorch_libdcll import *

n_epochs=5
n_iters = 100
in_channels = 1
out_channels_1 = 5
out_channels_2 = 5
im_width = 28
im_height = 28
batch_size=32
device = 'cpu'

layer1 = Conv2dDCLLlayer(in_channels, out_channels = out_channels_1).to(device)
#layer2 = DenseDCLLlayer(out_channels_1*im_width*im_height, out_channels = 100).to(device)

from load_mnist import *
gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = batch_size)
criterion = nn.MSELoss()
#optimizer = optim.SGD([layer1.i2h.weight, layer2.i2h.weight], lr=2e-5)
optimizer = optim.SGD([layer1.i2h.weight], lr=1e-2)
input, labels1h = gen_train.next()
input = torch.Tensor(input.swapaxes(1,3)).to(device)
labels1h = torch.Tensor(labels1h).to(device)
for epoch in range(n_epochs):
    states = []
    print(epoch)
    optimizer.zero_grad()
    layer1.zero_grad()
    #layer2.zero_grad()
    #init
    isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
    states.append(np.array(vmem1.clone()))
    #isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
    
    losses = 0 # For plotting
    for iter in range(n_iters):
        isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input,   isyn1, vmem1, eps01, eps11)
        if iter>50:
            losses += criterion(pvoutput1, labels1h)

    losses.backward()
    optimizer.step()
    print(layer1.i2h.weight)
    print(losses, float(torch.mean((pvoutput1.argmax(1) == labels1h.argmax(1)).float())))


