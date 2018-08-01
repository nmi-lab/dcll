#!/bin/python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : Sun 29 Jul 2018 01:37:45 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from pytorch_libdcll import *
from experimentTools import *
import time

def acc_mean(pvoutput, labels):
    return float(np.mean((np.array(pvoutput) == labels.argmax(2).cpu().numpy())))

def acc(pvoutput, labels):
    from collections import Counter
    pvoutput_ = np.array(pvoutput).T
    n = len(pvoutput_)
    arr = np.empty(n)
    arrl = np.empty(n)
    labels_ = labels.cpu().numpy().argmax(axis=2).T
    for i in range(n):
        arr[i] = Counter(pvoutput_[i]).most_common(1)[0][0]
        arrl[i] = Counter(labels_[i]).most_common(1)[0][0]
    return float(np.mean((arr == arrl)))

n_epochs=1000
n_iters = 600
in_channels = 2
out_channels_1 = 32
out_channels_2 = 64
out_channels_3 = 96
out_channels_4 = 96
out_channels_5 = 96
ds = 4
im_width =128//ds
im_height = 128//ds
batch_size = 32
output_size = 11
pad = 3
ksize = 7
burnin=300

layer1 = Conv2dDCLLlayer(in_channels,    out_channels = out_channels_1, im_width=im_width   , im_height=im_height   , output_size=output_size, pooling=2, padding=pad, kernel_size=ksize, alpha = .9, alphas = .9).to(device)
layer2 = Conv2dDCLLlayer(out_channels_1, out_channels = out_channels_2, im_width=im_width//2, im_height=im_height//2, output_size=output_size, pooling=2, padding=pad, kernel_size=ksize, alpha = .9, alphas = .9).to(device)
layer3 = Conv2dDCLLlayer(out_channels_2, out_channels = out_channels_3, im_width=im_width//4, im_height=im_height//4, output_size=output_size, pooling=1, padding=pad, kernel_size=ksize, alpha = .9, alphas = .9).to(device)
layer4 = Conv2dDCLLlayer(out_channels_3, out_channels = out_channels_4, im_width=im_width//4, im_height=im_height//4, output_size=output_size, pooling=1, padding=pad, kernel_size=ksize, alpha = .9, alphas = .9).to(device)
layer5 = Conv2dDCLLlayer(out_channels_4, out_channels = out_channels_5, im_width=im_width//4, im_height=im_height//4, output_size=output_size, pooling=1, padding=pad, kernel_size=ksize, alpha = .9, alphas = .9).to(device)

from load_dvsgestures_sparse import *
gen_train, gen_test = create_data(batch_size = batch_size, chunk_size = n_iters, size = [in_channels, im_width, im_height], ds = ds)
criterion = nn.MSELoss().to(device)
#optimizer = optim.Adam([layer1.i2h.weight, layer2.i2h.weight, layer3.i2h.weight] + [layer1.i2h.bias, layer2.i2h.bias, layer3.i2h.bias], lr=5e-6)
optimizer1 = optim.SGD([
    layer1.i2h.weight, layer1.i2h.bias,
    layer2.i2h.weight, layer2.i2h.bias], lr=1e-5)

optimizer2 = optim.SGD([
    layer3.i2h.weight, layer3.i2h.bias,
    layer4.i2h.weight, layer4.i2h.bias,
    layer5.i2h.weight, layer5.i2h.bias], lr=1e-5)

isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
isyn3, vmem3, eps03, eps13 = layer3.init_hiddens(batch_size)
isyn4, vmem4, eps04, eps14 = layer4.init_hiddens(batch_size)
isyn5, vmem5, eps05, eps15 = layer5.init_hiddens(batch_size)

acc_train = []
acc_test = []
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = .2e-5* (0.1 ** (epoch / n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    for epoch in range(n_epochs):
        input, labels1h = gen_train.next()
        input = torch.Tensor(input.swapaxes(0,1)).to(device).reshape(n_iters,batch_size,in_channels,im_width,im_height)
        labels1h = torch.Tensor(labels1h).to(device)

        clout1 = []
        clout2 = []
        clout3 = []
        clout4 = []
        clout5 = []
            
        cltout1 = []
        cltout2 = []
        cltout3 = []
        cltout4 = []
        cltout5 = []

     #   adjust_learning_rate(optimizer, epoch)

        for iter in range(n_iters):
            if iter>burnin:
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                layer1.zero_grad()
                layer2.zero_grad()
                layer3.zero_grad()
                layer4.zero_grad()
                layer5.zero_grad()

            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1,     isyn2, vmem2, eps02, eps12)
            isyn3, vmem3, eps03, eps13, output3, pvoutput3 = layer3.forward(output2,     isyn3, vmem3, eps03, eps13)
            isyn4, vmem4, eps04, eps14, output4, pvoutput4 = layer4.forward(output3,     isyn4, vmem4, eps04, eps14)
            isyn5, vmem5, eps05, eps15, output5, pvoutput5 = layer5.forward(output4,     isyn5, vmem5, eps05, eps15)
            if iter>=burnin:

                clout1.append(pvoutput1.argmax(1).detach().cpu().numpy())
                clout2.append(pvoutput2.argmax(1).detach().cpu().numpy())
                clout3.append(pvoutput3.argmax(1).detach().cpu().numpy())
                clout4.append(pvoutput4.argmax(1).detach().cpu().numpy())
                clout5.append(pvoutput5.argmax(1).detach().cpu().numpy())


                losses1 = criterion(pvoutput1, labels1h[iter])
                losses2 = criterion(pvoutput2, labels1h[iter])
                losses3 = criterion(pvoutput3, labels1h[iter])
                losses4 = criterion(pvoutput4, labels1h[iter])
                losses5 = criterion(pvoutput5, labels1h[iter])

                losses1.backward()
                losses2.backward()
                losses3.backward()
                losses4.backward()
                losses5.backward()

                optimizer1.step()
                optimizer2.step()
        #print("Step time: {0}".format(time.time()-t0))

        acc_train.append([
            acc(clout1,labels1h[burnin:]),
            acc(clout2, labels1h[burnin:]),
            acc(clout3, labels1h[burnin:]),
            acc(clout4, labels1h[burnin:]),
            acc(clout5, labels1h[burnin:])])
        print('TRAIN Epoch {0}: '.format(epoch))
        print(acc_train[-1])

        input_test, labels1h_test = gen_test.next()
        input_test = torch.Tensor(input_test.swapaxes(0,1)).to(device).reshape(n_iters,batch_size,in_channels,im_width,im_height)
        labels1h_test = torch.Tensor(labels1h_test).to(device)


        isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
        isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
        isyn3, vmem3, eps03, eps13 = layer3.init_hiddens(batch_size)
        isyn4, vmem4, eps04, eps14 = layer4.init_hiddens(batch_size)
        isyn5, vmem5, eps05, eps15 = layer5.init_hiddens(batch_size)

        #state = []
        #state.append(pvoutput5_test.detach().cpu().numpy())
        for iter in range(n_iters):
            isyn1, vmem1, eps01, eps11, output1, pvoutput1_test = layer1.forward(input_test[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2_test = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
            isyn3, vmem3, eps03, eps13, output3, pvoutput3_test = layer3.forward(output2, isyn3, vmem3, eps03, eps13)
            isyn4, vmem4, eps04, eps14, output4, pvoutput4_test = layer4.forward(output3, isyn4, vmem4, eps04, eps14)
            isyn5, vmem5, eps05, eps15, output5, pvoutput5_test = layer5.forward(output4, isyn5, vmem5, eps05, eps15)
            
            if iter>=burnin:
                cltout1.append(pvoutput1_test.argmax(1).detach().cpu().numpy())
                cltout2.append(pvoutput2_test.argmax(1).detach().cpu().numpy())
                cltout3.append(pvoutput3_test.argmax(1).detach().cpu().numpy())
                cltout4.append(pvoutput4_test.argmax(1).detach().cpu().numpy())
                cltout5.append(pvoutput5_test.argmax(1).detach().cpu().numpy())

        acc_test.append([
            acc(cltout1, labels1h_test[burnin:]),
            acc(cltout2, labels1h_test[burnin:]),
            acc(cltout3, labels1h_test[burnin:]),
            acc(cltout4, labels1h_test[burnin:]),
            acc(cltout5, labels1h_test[burnin:]),
            ])
        print('TEST  Epoch {0}: '.format(epoch))
        print(acc_test[-1])
    d = mksavedir()
    np.save(d+'/acc_train.npy', acc_train)
    np.save(d+'/acc_test.npy', acc_test)

