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

device = 'cpu'

class CLLDenseFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, weight, bias=None, alpha = .85, alphas=.8):
        isyn = alphas*prev_isyn + torch.mm(input, weight.t())
        vmem = alpha*prev_vmem + isyn
        eps0 = alphas*prev_eps0 + input 
        eps1 = alpha*prev_eps1 + eps0
        pv = torch.mm(eps1, weight.t())
        output = (vmem > .5).float()
        #ctx.save_for_backward(input, isyn, vmem, eps0, eps1, output, weight, bias)
        ctx.save_for_backward(input, pv, weight, bias)
        return isyn, vmem, eps0, eps1, output, pv

    @staticmethod
    def backward(ctx, *grad_output):
        #input, isyn, vmem, eps0, eps1, output, weight, bias = ctx.saved_tensors
        input, pv, weight, bias = ctx.saved_tensors
        grad_weights =  torch.mm(grad_output[-1].t(), input)
        #grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output[1], bias=bias, padding=2)
        return None, None, None, None, None, grad_weights, None, None, None

class CLLDenseModule(nn.Module):
    def __init__(self, in_channels, out_channels, bias=None, alpha = .9, alphas=.85):
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
        
    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        f = CLLDenseFunction.apply
        isyn, vmem, eps0, eps1, output, pv = f(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, self.weight, self.bias, self.alpha, self.alphas)
        return isyn, vmem, eps0, eps1, output, pv

    def init_prev(self, batch_size, im_width, im_height):
        return torch.zeros(batch_size, self.out_channels) 

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


class Conv2dDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, im_width=28, im_height=28, output_size=10):
        super(Conv2dDCLLlayer, self).__init__()
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
        pvoutput = self.softmax(torch.sigmoid(self.i2o(flatten)))
        return isyn, vmem, eps0, eps1, output, pvoutput

    def init_hiddens(self, batch_size):
        isyn = torch.zeros(batch_size, self.out_channels, self.im_width, self.im_height).to(device)
        vmem = torch.zeros(batch_size, self.out_channels, self.im_width, self.im_height).to(device)
        eps0 = torch.zeros(batch_size, self.in_channels , self.im_width, self.im_height).to(device)
        eps1 = torch.zeros(batch_size, self.in_channels , self.im_width, self.im_height).to(device)
        return isyn, vmem, eps0, eps1

class DenseDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, im_width=28, im_height=28, output_size=10):
        super(DenseDCLLlayer, self).__init__()
        self.im_width = im_width
        self.im_height = im_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.i2h = CLLDenseModule(in_channels,out_channels)
        self.i2o = nn.Linear(out_channels, output_size)
        self.i2o.weight.requires_grad = False
        self.i2o.bias.requires_grad = False
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        isyn, vmem, eps0, eps1, output, pv = self.i2h(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1)
        flatten = pv.view(-1,self.out_channels)
        pvoutput = self.softmax(torch.sigmoid(self.i2o(flatten)))
        return isyn, vmem, eps0, eps1, output, pvoutput

    def init_hiddens(self, batch_size):
        isyn = torch.zeros(batch_size, self.out_channels).to(device)
        vmem = torch.zeros(batch_size, self.out_channels).to(device)
        eps0 = torch.zeros(batch_size, self.in_channels ).to(device)
        eps1 = torch.zeros(batch_size, self.in_channels ).to(device)
        return isyn, vmem, eps0, eps1
                                                 


if __name__ == '__main__':
    #Test dense gradient
    f = CLLDenseFunction.apply




