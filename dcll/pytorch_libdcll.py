#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : Wed 1 Aug 2018 10:05
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.nn import functional as F
import numpy as np
from collections import namedtuple
import logging
from collections import Counter

def adjust_learning_rate(optimizer, epoch, base_lr = 5e-5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch / n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy_by_vote(pvoutput, labels):
    pvoutput_ = np.array(pvoutput).T
    n = len(pvoutput_)
    arr = np.empty(n)
    arrl = np.empty(n)
    labels_ = labels.cpu().numpy().argmax(axis=2).T
    for i in range(n):
        arr[i] = Counter(pvoutput_[i]).most_common(1)[0][0]
        arrl[i] = Counter(labels_[i]).most_common(1)[0][0]
    return float(np.mean((arr == arrl)))

def accuracy_by_mean(pvoutput, labels):
    return float(np.mean((np.array(pvoutput) == labels.argmax(2).cpu().numpy())))


# if gpu is to be used
device = 'cuda'

NeuronState = namedtuple(
    'NeuronState', ('isyn', 'vmem', 'eps0', 'eps1'))

class CLLDenseModule(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, alpha = .9, alphas=.85):
        super(CLLDenseModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = alpha
        self.alphas = alphas

    def reset_parameters(self):
        import math
        n = self.in_channels
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_state(self, batch_size, init_value = 0):
        self.state = NeuronState(
            isyn = torch.zeros(batch_size, self.out_channels).detach().to(device) + init_value,
            vmem = torch.zeros(batch_size, self.out_channels).detach().to(device) + init_value,
            eps0 = torch.zeros(batch_size, self.in_channels ).detach().to(device) + init_value,
            eps1 = torch.zeros(batch_size, self.in_channels ).detach().to(device) + init_value
            )
        return self

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.isyn.shape[0] == self.state.vmem.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logging.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.isyn.shape[0], input.shape[0]))
            self.init_state(input.shape[0])

        isyn = F.linear(input, self.weight, self.bias)
        isyn += self.alphas*self.state.isyn
        vmem = self.alpha*self.state.vmem + isyn
        eps0 = input + self.alphas*self.state.eps0
        eps1 = self.alpha*self.state.eps1 + eps0
        eps1 = eps1.detach()
        pv = torch.sigmoid(F.linear(eps1, self.weight, self.bias))
        output = (vmem > 0).float()
        # update the neuronal state
        self.state = NeuronState(isyn=isyn.detach(),
                                 vmem=vmem.detach(),
                                 eps0=eps0.detach(),
                                 eps1=eps1.detach())

        return output, pv

    #def init_prev(self, batch_size, im_width, im_height):
    #    return torch.zeros(batch_size, self.out_channels)

class DenseDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, target_size=None, alpha=.9, alphas = .85):
        super(DenseDCLLlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size
        self.i2h = CLLDenseModule(in_channels,out_channels, alpha=alpha, alphas=alphas)
        self.i2o = nn.Linear(out_channels, target_size, bias=False)
        self.i2o.weight.requires_grad = False
        # self.i2o.bias.requires_grad = False
        # self.softmax = nn.LogSoftmax(dim=1)
        self.input_size = self.out_channels
        self.init_dcll()

    def get_flat_size(self):
        return self.out_channels

    def forward(self, input):
        input     = input.view(-1,self.in_channels).detach()
        output, pv = self.i2h(input)
        pvoutput = self.i2o(pv)
        output = output.detach()
        return output, pvoutput, pv

    def init_hiddens(self, batch_size, init_value = 0):
        self.i2h.init_state(batch_size, init_value = init_value)
        return self

    def init_dcll(self):
        limit = np.sqrt(6.0 / (np.prod(self.out_channels) + self.target_size))
        self.M = torch.tensor(np.random.uniform(-limit, limit, size=[self.out_channels, self.target_size])).float()
        self.i2o.weight.data = self.M.t()
        limit = np.sqrt(1. / (np.prod(self.out_channels) + self.in_channels))
        # limit = np.sqrt(1e-32 / (np.prod(self.out_channels) + self.in_channels))
        self.i2h.weight.data = torch.tensor(np.random.uniform(-limit, limit, size=[self.in_channels, self.out_channels])).t().float()
        self.i2h.bias.data = torch.tensor(np.zeros([self.out_channels])).float()




class CLLConv2DModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1, bias=True, alpha = .95, alphas=.9):
        super(CLLConv2DModule, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = alpha
        self.alphas = alphas


    def reset_parameters(self):
        import math
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_state(self, batch_size, im_width, im_height, init_value = 0):
        dummy_input = torch.zeros(batch_size, self.in_channels, im_height, im_width).to(device)
        isyn_shape =  F.conv2d(dummy_input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).shape

        self.state = NeuronState(
            isyn = torch.zeros(isyn_shape).detach().to(device)+init_value,
            vmem = torch.zeros(isyn_shape).detach().to(device)+init_value,
            eps0 = torch.zeros(dummy_input.shape).detach().to(device)+init_value,
            eps1 = torch.zeros(dummy_input.shape).detach().to(device)+init_value
            )
        return self

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.isyn.shape[0] == self.state.vmem.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logging.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.isyn.shape[0], input.shape[0]))
            self.init_state(input.shape[0], input.shape[2], input.shape[3])

        isyn = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        isyn += self.alphas*self.state.isyn
        vmem = self.alpha*self.state.vmem + isyn
        eps0 = input + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0
        eps1 = eps1.detach()
        pv = torch.sigmoid(F.conv2d(eps1, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups))
        output = (vmem > 0).float()
        # update the neuronal state
        self.state = NeuronState(isyn=isyn.detach(),
                                 vmem=vmem.detach(),
                                 eps0=eps0.detach(),
                                 eps1=eps1.detach())
        return output, pv

    def init_prev(self, batch_size, im_width, im_height):
        return torch.zeros(batch_size, self.in_channels, im_width, im_height)


class Conv2dDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, im_width=28, im_height=28, target_size=10, pooling=1, padding = 2, alpha=.95, alphas=.9):
        super(Conv2dDCLLlayer, self).__init__()
        self.im_width = im_width
        self.im_height = im_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        if pooling>1:
            self.pooling = pooling
            self.pool = nn.MaxPool2d(kernel_size=pooling, stride=pooling, padding=0)
        else:
            self.pooling = pooling
            self.pool = lambda x: x
        self.kernel_size = kernel_size
        self.target_size = target_size
        self.i2h = CLLConv2DModule(in_channels,out_channels, kernel_size, padding=padding, alpha = alpha, alphas = alphas)
        self.i2o = nn.Linear(im_height*im_width*out_channels//pooling**2, target_size, bias=False)
        self.i2o.weight.requires_grad = False
        #self.i2o.bias.requires_grad = False
        #self.softmax = nn.LogSoftmax(dim=1)
        self.init_dcll()

    def get_flat_size(self):
        return int(self.im_height*self.im_width*self.out_channels//self.pooling**2)

    def forward(self, input):
        input     = input.detach()
        output, pv = self.i2h(input)
        pvp = self.pool(pv)
        flatten = pvp.view(-1, self.get_flat_size())
        pvoutput = self.i2o(flatten)
        output = output.detach()
        return self.pool(output), pvoutput, pv

    def init_hiddens(self, batch_size, init_value = 0):
        self.i2h.init_state(batch_size, self.im_height, self.im_width, init_value = init_value)
        return self

    def init_dcll(self):
        nh = self.get_flat_size()
        limit = np.sqrt(6.0 / (nh + self.target_size))
        self.M = torch.tensor(np.random.uniform(-limit, limit, size=[nh, self.target_size])).float()
        self.i2o.weight.data = self.M.t()
        limit = 1e-32
        # limit = .1
        self.i2h.weight.data = torch.tensor(np.random.uniform(-limit, limit, size=[self.out_channels, self.in_channels, self.kernel_size, self.kernel_size])).float()
        self.i2h.bias.data = torch.tensor(np.ones([self.out_channels])-1).float()

class DCLLBase(nn.Module):
    def __init__(self, dclllayer, batch_size=48, loss = torch.nn.MSELoss, optimizer = optim.SGD, kwargs_optimizer = {'lr':5e-5}, burnin = 200):
        super(DCLLBase, self).__init__()
        self.dclllayer = dclllayer
        self.crit = loss().to(device)
        self.optimizer = optimizer(dclllayer.i2h.parameters(), **kwargs_optimizer)
        self.burnin = burnin
        self.batch_size = batch_size
        self.init(self.batch_size)

    def init(self, batch_size, init_states = True):
        self.clout = []
        self.iter = 0
        if init_states: self.dclllayer.init_hiddens(batch_size, init_value = 0)

    def forward(self, input):
        self.iter+=1
        o, p, pv = self.dclllayer.forward(input)
        return o, p, pv

    def train(self, input, target):
        output, pvoutput, pv = self.forward(input)
        if self.iter>=self.burnin:
            self.optimizer.zero_grad()
            self.dclllayer.zero_grad()
            loss = self.crit(pvoutput, target) + 1e-6*torch.norm(pv,2)
            loss.backward()
            self.optimizer.step()
        return output, pvoutput, pv

class DCLLClassification(DCLLBase):
    def forward(self, input):
        o, p, pv = super(DCLLClassification, self).forward(input)
        self.clout.append(p.argmax(1).detach().cpu().numpy())
        return o,p,pv

    def accuracy(self, targets):
        cl = np.array(self.clout)
        begin = cl.shape[0]
        return accuracy_by_vote(cl, targets[-begin:])

class DCLLGeneration(DCLLBase):
    def init(self, batch_size, init_states = True):
        self.vmem_out = []
        self.spikes_out = []
        self.clout = []
        self.iter = 0
        if init_states: self.dclllayer.init_hiddens(batch_size, init_value = 0)
    def forward(self, input):
        o, p, pv = super(DCLLGeneration, self).forward(input)
        self.clout.append(p.detach().cpu().numpy())
        self.spikes_out.append(o.detach().cpu().numpy())
        self.vmem_out.append(self.dclllayer.i2h.state[1].detach().cpu().numpy())
        return o,p, pv



if __name__ == '__main__':
    #Test dense gradient
    f = CLLDenseFunction.apply
