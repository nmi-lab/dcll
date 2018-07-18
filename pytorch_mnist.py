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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import autograd
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np
from load_mnist import *

class CLLConv2DFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, weight, bias=None, stride=1, padding=2, dilation=1, groups=1, alpha = .9, alphas = .9):
        isyn = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        isyn += alphas*prev_isyn
        vmem = alpha*prev_vmem + isyn
        eps0 = input + alphas*prev_eps0
        eps1 = alpha*prev_eps1 + eps0
        pv = F.conv2d(eps1, weight, bias, stride, padding, dilation, groups)
        output = (vmem > .5).float()
        #ctx.save_for_backward(input, isyn, vmem, eps0, eps1, output, weight, bias)
        ctx.save_for_backward(input, pv, weight, bias)
        return isyn, vmem, eps0, eps1, output, pv

    @staticmethod
    def backward(ctx, *grad_output):
        #input, isyn, vmem, eps0, eps1, output, weight, bias = ctx.saved_tensors
        input, pv, weight, bias = ctx.saved_tensors
        grad_weights = nn.grad.conv2d_weight(input, weight.shape, grad_output[-1], bias=bias, padding=2)
        #grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output[1], bias=bias, padding=2)
        return None, None, None, None, None, grad_weights, None


class CLLConv2DModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1, bias=None, alpha = .9, alphas=.9):
        super(CLLConv2DModule, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
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
        
    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        f = CLLConv2DFunction.apply
        isyn, vmem, eps0, eps1, output, pv = f(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, self.weight, self.bias)
        return isyn, vmem, eps0, eps1, output, pv

    def init_prev(self, batch_size, im_width, im_height):
        return torch.zeros(batch_size, self.in_channels, im_width, im_height) 


class DCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, im_width=28, im_height=28, output_size=10):
        super(DCLLlayer, self).__init__()
        self.im_width = im_width
        self.im_height = im_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.i2h = CLLConv2DModule(in_channels,out_channels, kernel_size, padding=2)
        self.i2o = nn.Linear(im_height*im_width*out_channels, output_size)
        self.i2o.weight.requires_grad = False
        self.i2o.bias.requires_grad = False
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        isyn, vmem, eps0, eps1, output, pv = self.i2h(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1)
        flatten = pv.view(-1,self.im_height*self.im_width*self.out_channels)
        pvoutput = self.softmax(self.i2o(flatten))
        return isyn, vmem, eps0, eps1, output, pvoutput

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.out_channels, self.im_width, self. im_height) 

if __name__ == '__main__':
    n_epochs=200
    n_iters = 200
    in_channels = 1
    out_channels_1 = 5
    out_channels_2 = 5
    batch_size=32
    torch.cuda.set_device(0)

    with torch.cuda.device(0):
        layer1 = DCLLlayer(in_channels, out_channels = out_channels_1).cuda()
        layer2 = DCLLlayer(out_channels_1, out_channels = out_channels_2).cuda()

        gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = batch_size)
        criterion = nn.MSELoss()
        optimizer = optim.SGD([layer1.i2h.weight, layer2.i2h.weight], lr=2e-5)

        input, labels1h = gen_train.next()
        input = torch.Tensor(input.swapaxes(1,3)).cuda()
        labels1h = torch.Tensor(labels1h).cuda()
        for epoch in range(n_epochs):
            print(epoch)
            optimizer.zero_grad()
            layer1.zero_grad()
            layer2.zero_grad()
            #init
            isyn1 = torch.zeros(batch_size, out_channels_1, 28, 28).cuda()
            vmem1 = torch.zeros(batch_size, out_channels_1, 28, 28).cuda()
            eps01 = torch.zeros(batch_size, in_channels   , 28, 28).cuda()
            eps11 = torch.zeros(batch_size, in_channels   , 28, 28).cuda()
            isyn2 = torch.zeros(batch_size, out_channels_2, 28, 28).cuda()
            vmem2 = torch.zeros(batch_size, out_channels_2, 28, 28).cuda()
            eps02 = torch.zeros(batch_size, out_channels_1, 28, 28).cuda()
            eps12 = torch.zeros(batch_size, out_channels_1, 28, 28).cuda()
            
            losses = 0 # For plotting
            for iter in range(n_iters):
                isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input,   isyn1, vmem1, eps01, eps11)
                isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)
                losses += criterion(pvoutput1, labels1h)
                losses += criterion(pvoutput2, labels1h)

            losses.backward()
            optimizer.step()
            print(layer1.i2h.weight)
            print(losses, float(torch.mean((pvoutput1.argmax(1) == labels1h.argmax(1)).float())))
            print(losses, float(torch.mean((pvoutput2.argmax(1) == labels1h.argmax(1)).float())))

#if __name__ == '__other__':
#    f = CLLConv2DFunction.apply
#
#    gradient = torch.randn (batch_size, in_channels, 28,28)
#    model = CLLConv2DModule(1,20,5, padding=2)
#
#    weight = torch.rand([20,1,5,5], requires_grad = True)
#    bias = torch.rand([20], requires_grad = True)
#    isyn, vmem, eps0, eps1, output = f(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, weight, None)
#
#
#    c = torch.nn.Conv2d(1,20,5, padding=2, bias=None)
#    cc = c(input)
#    dc = cc.grad_fn(gradient, c.weight, c.bias, cc)
#
#

#
#criterion = nn.MSELoss()
#learning_rate = 0.0005
#


