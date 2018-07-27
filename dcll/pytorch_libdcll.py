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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch import autograd
import torchvision.transforms as transforms
from torch.nn import functional as F
import numpy as np

# if gpu is to be used
device = 'cuda:0'

# class CLLDenseFunction(autograd.Function):
#     @staticmethod
#     def forward(ctx, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, weight, bias=None, alpha = .95, alphas=.8):
#         isyn = alphas*prev_isyn + torch.addmm(bias,input, weight.t())
#         vmem = alpha*prev_vmem + isyn
#         eps0 = alphas*prev_eps0 + input
#         eps1 = alpha*prev_eps1 + eps0
#         pv = torch.addmm(bias, eps1, weight.t())
#         output = (torch.sigmoid(vmem) > .5).float()
#         #ctx.save_for_backward(input, isyn, vmem, eps0, eps1, output, weight, bias)
#         ctx.save_for_backward(input, pv, weight, bias)
#         return isyn, vmem, eps0, eps1, output, pv

#     @staticmethod
#     def backward(ctx, *grad_output):
#         #input, isyn, vmem, eps0, eps1, output, weight, bias = ctx.saved_tensors
#         input, pv, weight, bias = ctx.saved_tensors
#         grad_weights =  torch.mm(grad_output[-1].t(), input)
#         grad_bias =  torch.mm(grad_output[-1].t(), torch.ones_like(input)).sum(1)
#         #grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output[1], bias=bias, padding=2)
#         return None, None, None, None, None, grad_weights, grad_bias, None, None

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

    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        isyn = F.linear(input, self.weight, self.bias)
        isyn += self.alphas*prev_isyn
        vmem = self.alpha*prev_vmem + isyn
        eps0 = input + self.alphas*prev_eps0
        eps1 = self.alpha*prev_eps1 + eps0
        eps1 = eps1.detach()
        pv = F.linear(eps1, self.weight, self.bias)
        output = (vmem > 0).float()
        return isyn, vmem, eps0, eps1, output, pv

    def init_prev(self, batch_size, im_width, im_height):
        return torch.zeros(batch_size, self.out_channels)

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
        # self.softmax = nn.LogSoftmax(dim=1)
        self.init_dcll()

    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        input     = input    .detach()
        prev_isyn = prev_isyn.detach()
        prev_vmem = prev_vmem.detach()
        prev_eps0 = prev_eps0.detach()
        prev_eps1 = prev_eps1.detach()
        isyn, vmem, eps0, eps1, output, pv = self.i2h(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1)
        flatten = pv.view(-1,self.out_channels)
        pvoutput = torch.sigmoid(self.i2o(flatten))
        # pvoutput = self.softmax(self.i2o(flatten))
        return isyn, vmem, eps0, eps1, output, pvoutput

    def init_hiddens(self, batch_size):
        isyn = torch.zeros(batch_size, self.out_channels).to(device)
        vmem = torch.zeros(batch_size, self.out_channels).to(device)
        eps0 = torch.zeros(batch_size, self.in_channels ).to(device)
        eps1 = torch.zeros(batch_size, self.in_channels ).to(device)
        return isyn, vmem, eps0, eps1

    def init_dcll(self):
        limit = np.sqrt(6.0 / (np.prod(self.out_channels) + self.output_size))
        self.M = torch.tensor(np.random.uniform(-limit, limit, size=[self.out_channels, self.output_size])).float()
        self.i2o.weight.data = self.M.t()
        limit = np.sqrt(1e-32 / (np.prod(self.out_channels) + self.in_channels))
        self.i2h.weight.data = torch.tensor(np.random.uniform(-limit, limit, size=[self.in_channels, self.out_channels])).t().float()
        self.i2h.bias.data = torch.tensor(np.ones([self.out_channels])-1).float()


#class CLLConv2DFunction(autograd.Function):
#    @staticmethod
#    def forward(ctx, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1, weight, bias=None, stride=1, padding=2, dilation=1, groups=1, alpha = .9, alphas = .9):
#        isyn = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
#        isyn += alphas*prev_isyn
#        vmem = alpha*prev_vmem + isyn
#        eps0 = input + alphas*prev_eps0
#        eps1 = alpha*prev_eps1 + eps0
#        pv = F.conv2d(eps1, weight, bias, stride, padding, dilation, groups)
#        output = (vmem > .75).float()
#        #ctx.save_for_backward(input, isyn, vmem, eps0, eps1, output, weight, bias)
#        ctx.save_for_backward(input, pv, weight, bias)
#        return isyn, vmem, eps0, eps1, output, pv
#
#    @staticmethod
#    def backward(ctx, *grad_output):
#        #input, isyn, vmem, eps0, eps1, output, weight, bias = ctx.saved_tensors
#        input, pv, weight, bias = ctx.saved_tensors
#        grad_weights = nn.grad.conv2d_weight(input, weight.shape, grad_output[-1], bias=bias, padding=2)
#        #grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output[1], bias=bias, padding=2)
#        return None, None, None, None, None, grad_weights, None
#

class CLLConv2DModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1, bias=True, alpha = .95, alphas=.9, pooling = 1):
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
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        isyn = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        isyn += self.alphas*prev_isyn
        vmem = self.alpha*prev_vmem + isyn
        eps0 = input + self.alphas*prev_eps0
        eps1 = self.alpha*prev_eps1 + eps0
        eps1 = eps1.detach()
        pv = F.conv2d(eps1, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = (vmem > 0).float()
        return isyn, vmem, eps0, eps1, output, pv

    def init_prev(self, batch_size, im_width, im_height):
        return torch.zeros(batch_size, self.in_channels, im_width, im_height)


class Conv2dDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, im_width=28, im_height=28, output_size=10, pooling=1, padding = 2):
        super(Conv2dDCLLlayer, self).__init__()
        self.im_width = im_width
        self.im_height = im_height
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.pool = nn.MaxPool2d(pooling, stride=pooling)
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.i2h = CLLConv2DModule(in_channels,out_channels, kernel_size, padding=padding)
        self.i2o = nn.Linear(im_height*im_width*out_channels//pooling**2, output_size)
        self.i2o.weight.requires_grad = False
        self.i2o.bias.requires_grad = False
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_dcll()

    def forward(self, input, prev_isyn, prev_vmem, prev_eps0, prev_eps1):
        input     = input.detach()
        pooling = self.pooling
        prev_isyn = prev_isyn.detach()
        prev_vmem = prev_vmem.detach()
        prev_eps0 = prev_eps0.detach()
        prev_eps1 = prev_eps1.detach()
        isyn, vmem, eps0, eps1, output, pv = self.i2h(input, prev_isyn, prev_vmem, prev_eps0, prev_eps1)
        flatten = self.pool(torch.sigmoid(pv)).view(-1,self.im_height*self.im_width*self.out_channels//pooling**2)
        pvoutput = torch.sigmoid(self.i2o(flatten))
        output = output.detach()
        return isyn, vmem, eps0, eps1, self.pool(output), pvoutput

    def init_hiddens(self, batch_size):
        isyn = torch.zeros(batch_size, self.out_channels, self.im_width, self.im_height).to(device)
        vmem = torch.zeros(batch_size, self.out_channels, self.im_width, self.im_height).to(device)
        eps0 = torch.zeros(batch_size, self.in_channels , self.im_width, self.im_height).to(device)
        eps1 = torch.zeros(batch_size, self.in_channels , self.im_width, self.im_height).to(device)
        return isyn, vmem, eps0, eps1

    def init_dcll(self):
        pooling = self.pooling
        nh = int(self.im_height*self.im_width*self.out_channels//pooling**2)
        limit = np.sqrt(6.0 / (nh + self.output_size))
        self.M = torch.tensor(np.random.uniform(-limit, limit, size=[nh, self.output_size])).float()
        self.i2o.weight.data = self.M.t()
        limit = 1e-32
        self.i2h.weight.data = torch.tensor(np.random.uniform(-limit, limit, size=[self.out_channels, self.in_channels, self.kernel_size, self.kernel_size])).float()
        self.i2h.bias.data = torch.tensor(np.ones([self.out_channels])-1).float()




if __name__ == '__main__':
    #Test dense gradient
    f = CLLDenseFunction.apply
