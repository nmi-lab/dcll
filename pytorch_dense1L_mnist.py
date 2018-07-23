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
    n_epochs=200
    n_iters = 500
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
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer1 = optim.SGD([layer1.i2h.weight, layer1.i2h.bias], lr=1e-5)
    optimizer2 = optim.SGD([layer2.i2h.weight, layer2.i2h.bias], lr=1e-5)

    for epoch in range(n_epochs):
        input, labels1h = image2spiketrain(*gen_train.next())
        input = torch.Tensor(input).to(device)
        labels1h = torch.Tensor(labels1h).to(device)
        states1 = []
        states2 = []

        isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
        isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
        
        for iter in range(n_iters):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            layer1.zero_grad()
            layer2.zero_grad()
            isyn1 = isyn1.detach()
            vmem1 = vmem1.detach()
            eps01 = eps01.detach()
            eps11 = eps11.detach()
            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)

            isyn2 = isyn2.detach()
            vmem2 = vmem2.detach()
            eps02 = eps02.detach()
            eps12 = eps12.detach()
            output1 = output1.detach()

            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)

            states1.append(np.array(output1.detach().cpu().numpy()))
            states2.append(np.array(output2.detach().cpu().numpy()))
            if iter>50:
                losses1 = criterion1(pvoutput1, labels1h[-1])
                losses2 = criterion2(pvoutput2, labels1h[-1])

                losses1.backward()
                losses2.backward()
                optimizer1.step()
                optimizer2.step()
        #print('Epoch {0}: L1 {1:1.3}  L2 {2:1.3} Acc1 {3:1.3} Acc2 {4:1.3}'.format(epoch, losses1.cpu(), losses2.cpu(), acc(pvoutput1,labels1h), acc(pvoutput2, labels1h)))
        a = np.array(states1)
        b = np.array(states2)

        input, labels1h = image2spiketrain(*gen_train.next())
        input = torch.Tensor(input).to(device)
        labels1h = torch.Tensor(labels1h).to(device)

        for iter in range(n_iters):
            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)

        print('Test Epoch {0}: L1 {1:1.3}  L2 {2:1.3} Acc1 {3:1.3} Acc2 {4:1.3}'.format(epoch, losses1.cpu(), losses2.cpu(), acc(pvoutput1,labels1h), acc(pvoutput2, labels1h)))
        a = np.array(states1)
        b = np.array(states2)

