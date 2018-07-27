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
n_iters = 500
in_channels = 1
out_channels_1 = 32
out_channels_2 = 48
out_channels_3 = 64
im_width = 76
im_height = 60
batch_size = 64
output_size = 11

layer1 = Conv2dDCLLlayer(in_channels,    out_channels = out_channels_1, im_width=im_width  , im_height=im_height  , output_size=output_size, pooling=2, padding=3, kernel_size=7).to(device)
layer2 = Conv2dDCLLlayer(out_channels_1, out_channels = out_channels_2, im_width=im_width/2, im_height=im_height/2, output_size=output_size, pooling=1, padding=3, kernel_size=7).to(device)
layer3 = Conv2dDCLLlayer(out_channels_2, out_channels = out_channels_3, im_width=im_width/2, im_height=im_height/2, output_size=output_size, pooling=1, padding=3, kernel_size=7).to(device)

from load_massiset import *
gen_train, gen_test = create_data(valid=False, batch_size = batch_size, chunk_size = n_iters)
criterion = nn.MSELoss().to(device)
#optimizer = optim.Adam([layer1.i2h.weight, layer2.i2h.weight, layer3.i2h.weight] + [layer1.i2h.bias, layer2.i2h.bias, layer3.i2h.bias], lr=5e-6)
optimizer = optim.SGD([layer1.i2h.weight, layer2.i2h.weight, layer3.i2h.weight] + [layer1.i2h.bias, layer2.i2h.bias, layer3.i2h.bias], lr=5e-5)

isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
isyn3, vmem3, eps03, eps13 = layer3.init_hiddens(batch_size)



acc_train = []
acc_test = []
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 5e-5* (0.1 ** (epoch / n_epochs))
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
            
        cltout1 = []
        cltout2 = []
        cltout3 = []

        adjust_learning_rate(optimizer, epoch)

        

        for iter in range(n_iters):
            if iter>150:
                optimizer.zero_grad()
                layer1.zero_grad()
                layer2.zero_grad()
                layer3.zero_grad()

            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
            isyn3, vmem3, eps03, eps13, output3, pvoutput3 = layer3.forward(output2, isyn3, vmem3, eps03, eps13)
            if iter>=150:

                clout1.append(pvoutput1.argmax(1).detach().cpu().numpy())
                clout2.append(pvoutput2.argmax(1).detach().cpu().numpy())
                clout3.append(pvoutput3.argmax(1).detach().cpu().numpy())


                losses1 = criterion(pvoutput1, labels1h[iter])
                losses2 = criterion(pvoutput2, labels1h[iter])
                losses3 = criterion(pvoutput3, labels1h[iter])

                losses1.backward()
                losses2.backward()
                losses3.backward()

                optimizer.step()
        #print("Step time: {0}".format(time.time()-t0))
        acc_train.append([acc(clout1,labels1h[150:]), acc(clout2, labels1h[150:]), acc(clout3, labels1h[150:])])
        print('TRAIN Epoch {0}: Acc1 {1:1.3} Acc2 {2:1.3} Acc3 {3:1.3}'.format(epoch, *acc_train[-1]))

        input_test, labels1h_test = gen_test.next()
        input_test = torch.Tensor(input_test.swapaxes(0,1)).to(device).reshape(n_iters,batch_size,in_channels,im_width,im_height)
        labels1h_test = torch.Tensor(labels1h_test).to(device)

        for iter in range(n_iters):
            isyn1, vmem1, eps01, eps11, output1, pvoutput1_test = layer1.forward(input_test[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2_test = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
            isyn3, vmem3, eps03, eps13, output3, pvoutput3_test = layer3.forward(output2, isyn3, vmem3, eps03, eps13)
            
            if iter>=250:
                cltout1.append(pvoutput1_test.argmax(1).detach().cpu().numpy())
                cltout2.append(pvoutput2_test.argmax(1).detach().cpu().numpy())
                cltout3.append(pvoutput3_test.argmax(1).detach().cpu().numpy())



            #if (iter%100)==0:
            #    print(mapping2[int(pvoutput3_test.argmax(1).detach().cpu().numpy())])

        acc_test.append([acc(cltout1, labels1h_test[250:]), acc(cltout2, labels1h_test[250:]), acc(cltout3, labels1h_test[250:])])
        print('TEST  Epoch {0}: Acc1 {1:1.3} Acc2 {2:1.3} Acc3 {3:1.3}'.format(epoch, *acc_test[-1]))
    #a = np.array(state1)
    #b = np.array(state2)
    #c = np.array(state3)

    np.save('Results/006/acc_train.npy', acc_train)
    np.save('Results/006/acc_test.npy', acc_test)
    np.save('Results/006/a.npy', a)
    np.save('Results/006/b.npy', b)
    #np.save('Results/006/c.npy', c)

