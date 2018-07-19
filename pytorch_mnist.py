#!/bin/python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from pytorch_libdcll import *

def acc(pvoutput, labels):
    return float(torch.mean((pvoutput.argmax(1) == labels[-1].argmax(1)).float()))

if __name__ == '__main__':
    n_epochs=20
    n_iters = 200
    in_channels = 1
    out_channels_1 = 100
    out_channels_2 = 100
    im_width = 28
    im_height = 28
    batch_size = 32

    layer1 = DenseDCLLlayer(im_width*im_height, out_channels = out_channels_1).to(device)
    layer2 = DenseDCLLlayer(out_channels_1, out_channels = out_channels_2).to(device)

    from load_mnist import *
    gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([layer1.i2h.weight, layer2.i2h.weight], lr=5e-4)



    for epoch in range(n_epochs):
        input, labels1h = image2spiketrain(*gen_train.next())
        input = torch.Tensor(input).to(device)
        labels1h = torch.Tensor(labels1h).to(device)
        states1 = []
        states2 = []
        optimizer.zero_grad()
        layer1.zero_grad()
        layer2.zero_grad()
        isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
        isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
        
        losses = 0 # For plotting
        for iter in range(n_iters):
            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
            states1.append(np.array(vmem1.detach().numpy()))
            states2.append(np.array(vmem2.detach().numpy()))
            if iter>50:
                losses += criterion(pvoutput1, labels1h[-1])
                losses += criterion(pvoutput2, labels1h[-1])

        losses.backward()
        optimizer.step()
        print(layer2.i2h.weight)
        print(epoch, losses, acc(pvoutput1,labels1h), acc(pvoutput2, labels1h))

