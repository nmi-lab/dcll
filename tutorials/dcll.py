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
from itertools import chain
import dclllib
import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import pylab as plt
from collections import namedtuple
    
device = 'cuda'
dtype = torch.float32

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >= 0).float()

    def backward(aux, grad_output):
        #grad_input = grad_output.clone()        
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input
    
smooth_step = SmoothStep().apply

class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    def __init__(self, layer, alpha = .9, beta=.85):
        super(LIFLayer, self).__init__()        
        self.layer = layer
        self.in_channels = layer.in_features
        self.out_channels = layer.out_features
        self.alpha = alpha
        self.beta = beta
        self.state = self.NeuronState(P=torch.zeros(self.in_channels).type(dtype),
                                      Q=torch.zeros(self.in_channels).type(dtype),
                                      R=torch.zeros(self.out_channels).type(dtype),
                                      S=torch.zeros(self.out_channels).type(dtype))
        self.layer.weight.data.uniform_(-.3, .3)
        self.layer.bias.data.uniform_(-.01, .01)        
        
    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = self.NeuronState(P=self.state.P.cuda(device),
                                      Q=self.state.Q.cuda(device),
                                      R=self.state.R.cuda(device),
                                      S=self.state.S.cuda(device))
        self.layer = self.layer.cuda()
        return self 
    
    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = self.NeuronState(P=self.state.P.cpu(device),
                                      Q=self.state.Q.cpu(device),
                                      R=self.state.R.cpu(device),
                                      S=self.state.S.cpu(device))
        self.layer = self.layer.cpu()
        return self 
    
    def forward(self, Sin_t):
        state = self.state
        P = self.alpha*state.P + state.Q
        R = self.alpha*state.R - state.S
        Q = self.beta*state.Q + Sin_t
        U = self.layer(P) + R
        # update the neuronal state
        S = smooth_step(U)
        self.state = self.NeuronState(P=P.detach(), Q=Q.detach(), R=R.detach(), S=S.detach())
        return self.state, U
    
device = 'cpu'

class DCLL(nn.Module):
    def __init__(self, Nin, Nhid=128, Nout=1):
        super(DCLL, self).__init__()        
        self.layer1 = LIFLayer(nn.Linear(Nin, Nhid))
        self.le_layer1 = nn.Linear(Nhid, Nout) #lt_ = locally trainable
        self.layer2 = LIFLayer(nn.Linear(Nhid, Nhid))
        self.le_layer2 = nn.Linear(Nhid, Nout) #lt_ = locally trainable
        self.layer3 = LIFLayer(nn.Linear(Nhid, Nhid))
        self.le_layer3 = nn.Linear(Nhid, Nout) #lt_ = locally trainable
        self.layers = [self.layer1, self.layer2, self.layer3]
        self.drop_out = nn.Dropout(.5)
        
    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        [l.cuda() for l in self.layers]
        return self 
    
    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        [l.cpu() for l in self.layers]
        return self 
    
    def __len__(self):
        return len(self.layers)
    
    def forward(self, input):
        s1, u1 = self.layer1(input)
        s1dp = s1[-1] #self.drop_out(s1[-1])
        r1 = self.le_layer1(smooth_step(u1))
        
        s2, u2 = self.layer2(s1dp) 
        s2dp = s2[-1] #self.drop_out(s2[-1])
        r2 = self.le_layer2(smooth_step(u2)) 
        
        s3, u3 = self.layer3(s2dp) 
        s3dp = s3[-1] #self.drop_out(s2[-1])
        r3 = self.le_layer3(smooth_step(u3))
        
        return [s1,s2,s3],[r1,r2,r3]
    
    def get_trainable_parameters(self):
        return chain(*[l.parameters() for l in self.layers])
    
    

if __name__=="__main__":
    Nout = 128
    Nin = 100
    T = 400
    Sin = torch.FloatTensor(dclllib.spiketrains(N=Nin, T=T, rates = np.ones([Nin])*25))
    Sin = Sin.to(device)
    
    #Set up targets
    yhat = np.zeros(T)
    yhat[100]=1; yhat[200]=1; yhat[300]=1; 
    yhat = np.convolve(yhat,np.exp(-np.linspace(0,1,100)/.1))
    yhat_t = torch.FloatTensor(yhat).to(device)

    #Set up Network Layers
    net = DCLL(Nin).to(device)
    mse_loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(net.get_trainable_parameters(), lr=1e-4, betas=[0., .95])
    
    for e in range(300):    
        loss_hist = 0
        for n in range(T):
            s, r = net.forward(Sin[n])
            loss_tv = 0
            for i in range(len(net)): loss_tv += mse_loss(r[i],yhat_t[n]) + .1*torch.mean(s[i][-1])
            loss_tv.backward()
            opt.step()
            opt.zero_grad()
            loss_hist += loss_tv
        if (e%5)==0: print(e, loss_hist)
            
    Sprobe = np.empty([len(net),T,Nout])
    readProbe = np.empty([len(net),T,1])
    for n in range(T):
        s,r = net.forward(Sin[n])
        for i in range(len(net)):
            Sprobe[i,n] = s[i][-1].clone().data.cpu().numpy()
            readProbe[i,n] = r[i].data.cpu().numpy()
            
    for i in range(len(net)):
        plt.figure()
        ax1, ax2 = dclllib.plotLIF(U=readProbe[i], S=Sprobe[i])
    

