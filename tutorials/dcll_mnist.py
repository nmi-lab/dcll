#!/bin/python
#-----------------------------------------------------------------------------
# File Name : multilayer.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 12-03-2019
# Last Modified : Tue 12 Mar 2019 04:51:44 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from dcll import *
    
device = 'cuda'
dtype = torch.float32

gen_train = dclllib.get_mnist_loader(100, Nparts=60, train=True)
gen_test = dclllib.get_mnist_loader(100, Nparts=10, train=False)

def iter_mnist(gen_train, batchsize=100, T=1000, max_rate = 100):
    datait = iter(gen_train)
    for raw_input, raw_labels in datait:
        data, labels1h = dclllib.image2spiketrain(raw_input, raw_labels, max_duration=T, gain=max_rate)
        data_t = torch.FloatTensor(data).view(T,batchsize,-1)
        labels_t = torch.Tensor(labels1h)
        yield data_t, labels_t



if __name__=="__main__":
    Nout = 128
    Nin = 100
    T = 300
    
    #Set up Network Layers
    net = DCLL(Nin = 784, Nout=10, Nhid = 128).to(device)
    opt = torch.optim.Adam(net.get_trainable_parameters(), lr=1e-4, betas=[0., .95])
    mse_loss = torch.nn.MSELoss()

    
    data, target = next(iter_mnist(gen_train, T=T))
    net.init(data[0].to(device))

    def dcll_loss(r, s, tgt):
        loss_tv = 0
        for i in range(len(r)):
            loss_tv += mse_loss(r[i],tgt) + .1*torch.mean(s[i][-1])
        return loss_tv
    
    for e in range(T):    
        loss_hist = 0
        for data, label in iter_mnist(gen_train, T=T):
            data_d = data.to(device)
            label_d = label.to(device)
            for n in range(T):
                st, rt = net.forward(data_d[n])        
                loss_tv = dcll_loss(rt, st, label_d[n])
                loss_tv.backward()
                opt.step()
                opt.zero_grad()
            loss_hist += loss_tv
        print(e, loss_hist)
    

