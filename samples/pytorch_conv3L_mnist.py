#!/usr/bin/env python
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
from dcll.pytorch_libdcll import *
import time

def acc(pvoutput, labels):
    return float(torch.mean((pvoutput.argmax(1) == labels[-1].argmax(1)).float()))

if __name__ == '__main__':
    n_epochs=500
    n_iters = 500
    in_channels = 1
    out_channels_1 = 32//2
    out_channels_2 = 48//2
    out_channels_3 = 64//2
    im_width = 28
    im_height = 28
    batch_size = 64
    output_size = 10

    layer1 = Conv2dDCLLlayer(in_channels,    out_channels = out_channels_1, im_width=im_width  , im_height=im_height  , output_size=output_size, pooling=2, padding=3, kernel_size=7).to(device)
    layer2 = Conv2dDCLLlayer(out_channels_1, out_channels = out_channels_2, im_width=im_width/2, im_height=im_height/2, output_size=output_size, pooling=2, padding=3, kernel_size=7).to(device)
    layer3 = Conv2dDCLLlayer(out_channels_2, out_channels = out_channels_3, im_width=im_width/4, im_height=im_height/4, output_size=output_size, pooling=1, padding=3, kernel_size=7).to(device)

    from dcll.load_mnist import *
    gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = batch_size)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD([layer1.i2h.weight, layer2.i2h.weight, layer3.i2h.weight] + [layer1.i2h.bias, layer2.i2h.bias, layer3.i2h.bias], lr=5e-5)

    isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
    isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
    isyn3, vmem3, eps03, eps13 = layer3.init_hiddens(batch_size)


    input_test, labels1h_test = image2spiketrain(*gen_train.next())
    input_test = torch.Tensor(input_test).to(device).reshape(n_iters,batch_size,in_channels,im_width,im_height)
    labels1h_test = torch.Tensor(labels1h_test).to(device)



    for epoch in range(n_epochs):
        input, labels1h = image2spiketrain(*gen_train.next())
        input = torch.Tensor(input).to(device).reshape(n_iters,batch_size,in_channels,im_width,im_height)
        labels1h = torch.Tensor(labels1h).to(device)



        for iter in range(n_iters):
            if iter>150:
                optimizer.zero_grad()
                layer1.zero_grad()
                layer2.zero_grad()
                layer3.zero_grad()

            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
            isyn3, vmem3, eps03, eps13, output3, pvoutput3 = layer3.forward(output2, isyn3, vmem3, eps03, eps13)

            if iter>150:
                losses1 = criterion(pvoutput1, labels1h[-1])
                losses2 = criterion(pvoutput2, labels1h[-1])
                losses3 = criterion(pvoutput3, labels1h[-1])

                losses1.backward()
                losses2.backward()
                losses3.backward()

                optimizer.step()
        #print("Step time: {0}".format(time.time()-t0))
        print(output1.detach().cpu().numpy().mean(), output2.detach().cpu().numpy().mean(), output3.detach().cpu().numpy().mean())
        print('TRAIN Epoch {0}: Acc1 {1:1.3} Acc2 {2:1.3} Acc3 {3:1.3}'.format(epoch, acc(pvoutput1,labels1h), acc(pvoutput2, labels1h), acc(pvoutput3, labels1h)))

        state1 = []
        for iter in range(n_iters-100):
            isyn1, vmem1, eps01, eps11, output1, pvoutput1_test = layer1.forward(input_test[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2_test = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
            isyn3, vmem3, eps03, eps13, output3, pvoutput3_test = layer3.forward(output2, isyn3, vmem3, eps03, eps13)

        print('TEST  Epoch {0}: Acc1 {1:1.3} Acc2 {2:1.3} Acc3 {3:1.3}'.format(epoch, acc(pvoutput1_test,labels1h_test), acc(pvoutput2_test, labels1h_test), acc(pvoutput3_test, labels1h_test)))
        #a = np.array(states1)
        #b = np.array(states2)
