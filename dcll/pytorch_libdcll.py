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
from torch import autograd
from torch.nn import functional as F
import numpy as np
from collections import namedtuple
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# if gpu is to be used
device = 'cuda'

NeuronState = namedtuple(
    'NeuronState', ['isyn', 'vmem', 'eps0', 'eps1'])

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


reducedNeuronState = namedtuple(
    'NeuronState', ('eps0', 'eps1', 'arp'))

class CLLDenseModule(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, alpha = .9, alphas=.85, act = nn.Sigmoid()):
        super(CLLDenseModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = alpha
        self.alphas = alphas
        self.act = act
        # self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        # self.alphas = torch.nn.Parameter(torch.Tensor([alphas]))
        # self.alphas = torch.nn.Parameter(torch.ones(self.out_channels) * alphas)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
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
        return self.state

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.isyn.shape[0] == self.state.vmem.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                           .format(self.state.isyn.shape[0], input.shape[0]))
            self.init_state(input.shape[0])

        # clamp alphas to [0,1] range
        # self.alpha.data = self.alpha.clamp(0., 1.)
        # self.alphas.data = self.alphas.clamp(0., 1.)

        isyn = F.linear(input, self.weight, self.bias)
        isyn += self.alphas*self.state.isyn
        vmem = self.alpha*self.state.vmem + isyn
        eps0 = input + self.alphas*self.state.eps0
        eps1 = self.alpha*self.state.eps1 + eps0
        pv = self.act(F.linear(eps1, self.weight, self.bias))
        output = (vmem > 0).float()
        # update the neuronal state
        self.state = NeuronState(isyn=isyn.detach(),
                                 vmem=vmem.detach(),
                                 eps0=eps0.detach(),
                                 eps1=eps1.detach())

        return output, pv

class CLLDenseRRPModule(CLLDenseModule):
    def __init__(self, in_channels, out_channels, bias=True, alpha = .95, alphas=.9, alpharp = .65, wrp = 100, act = nn.Sigmoid()):
        super(CLLDenseRRPModule, self).__init__(in_channels, out_channels, bias, alpha, alphas, act)
        self.wrp=wrp
        self.alpharp=alpharp

    def init_state(self, batch_size, init_value = 0):
        self.state = reducedNeuronState(
            eps0 = torch.zeros(batch_size, self.in_channels ).detach().to(device) + init_value,
            eps1 = torch.zeros(batch_size, self.in_channels ).detach().to(device) + init_value,
            arp = torch.zeros(batch_size, self.out_channels).detach().to(device) + init_value,
        )
        return self.state

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0])

        eps0 = input + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + self.state.eps0
        pvmem = F.linear(eps1, self.weight, self.bias) - self.state.arp
        pv = self.act(pvmem)
        output = (pvmem>0).float()
        ##best
        #arp = .65*self.state.arp + output*10
        arp = self.alpharp*self.state.arp + output*self.wrp
        self.state = reducedNeuronState(
                         eps0=eps0.detach(),
                         eps1=eps1,
                         arp=arp)
        return output, pv

class DenseDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, target_size=None, bias= True, alpha=.9, alphas = .85, alpharp =.65, wrp = 0., act = nn.Sigmoid()):
        if (target_size is None):
            target_size = out_channels
        super(DenseDCLLlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_size = target_size
        if wrp>0:
            self.i2h = CLLDenseRRPModule(in_channels,out_channels, alpha = alpha, alphas = alphas, alpharp = alpharp, wrp = wrp, bias = bias, act = act)
        else:
            self.i2h = CLLDenseModule(in_channels,out_channels, alpha=alpha, alphas=alphas, bias = bias, act = act)
        self.i2o = nn.Linear(out_channels, target_size, bias=True)
        self.i2o.weight.requires_grad = False
        self.i2o.bias.requires_grad = False
        # self.softmax = nn.LogSoftmax(dim=1)
        self.input_size = self.out_channels

    def forward(self, input):
        input   = input.view(-1,self.in_channels)
        output, pv = self.i2h(input)
        pvoutput = self.i2o(pv)
        output = output.detach()
        return output, pvoutput, pv

    def init_hiddens(self, batch_size, init_value = 0):
        self.i2h.init_state(batch_size, init_value = init_value)
        return self

    def reset_tracks(self, mask = None):
        if mask is None:
            self.init_hiddens(self.i2h.state.isyn.shape[0])
            return
        for field in self.i2h.state:
            field[mask] = 0.

class CLLConv2DModule(nn.Module):
    NeuronState = namedtuple(
    'NeuronState', ('eps0', 'eps1'))
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1, bias=True, alpha = .95, alphas=.9, act = nn.Sigmoid()):
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
        self.act = act

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = alpha
        self.alphas = alphas


    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_output_shape(self, im_dims):
        dummy_input = torch.zeros(1, self.in_channels, im_dims[0], im_dims[1])
        if self.weight.is_cuda:
            dummy_input = dummy_input.to(device)
        out_shape =  F.conv2d(dummy_input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).shape
        return out_shape[1:] # remove batch_size

    def init_state(self, batch_size, im_dims, init_value = 0):
        input_shape = [batch_size, self.in_channels, im_dims[0], im_dims[1]]
        isyn_shape =  torch.Size([batch_size]) + self.get_output_shape(im_dims)

        self.state = self.NeuronState(
            eps0 = torch.zeros(input_shape).detach().to(device)+init_value,
            eps1 = torch.zeros(input_shape).detach().to(device)+init_value
            )
        return self.state

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logging.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))

            self.init_state(input.shape[0], input.shape[2], input.shape[3])

       # isyn = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
       # isyn += self.alphas*self.state.isyn
       # vmem = self.alpha*self.state.vmem + isyn
        eps0 = input + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0
        pvmem = F.conv2d(eps1, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = (pvmem>0).float()
        pv = self.act(pvmem)

        ##best
        #arp = .65*self.state.arp + output*10
        self.state = self.NeuronState( eps0=eps0.detach(),
                                       eps1=eps1.detach())
        return output, pv

    def init_prev(self, batch_size, im_dims):
        return torch.zeros(batch_size, self.in_channels, im_dims[0], im_dims[1])

class CLLConv2DRRPModule(CLLConv2DModule):
    NeuronState = namedtuple(
    'NeuronState', ('eps0', 'eps1', 'arp'))

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=2, dilation=1, groups=1, bias=True, alpha = .95, alphas=.9, alpharp = .65, wrp = 100, act = nn.Sigmoid()):
        '''
        Continuous local learning with relative refractory period. No isyn or vmem dynamics for speed and memory.
        *wrp*: weight for the relative refractory period
        '''
        super(CLLConv2DRRPModule, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, alpha, alphas, act)

        ##best
        #self.tarp=10
        self.wrp=wrp
        self.alpharp=alpharp
        self.iter=0

    def init_state(self, batch_size, im_dims, init_value = 0):
        input_shape = [batch_size, self.in_channels, im_dims[0], im_dims[1]]
        isyn_shape =  torch.Size([batch_size]) + self.get_output_shape(im_dims)


        self.state = reducedNeuronState(
            eps0 = torch.zeros(input_shape).detach().to(device)+init_value,
            eps1 = torch.zeros(input_shape).detach().to(device)+init_value,
            arp = torch.zeros(isyn_shape).detach().to(device)+init_value
            )
        return self.state

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0], input.shape[2], input.shape[3])

        eps0 = input + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0
        pvmem = F.conv2d(eps1, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups) - self.state.arp
        output = (pvmem>0).float()
        pv = self.act(pvmem)

        ##best
        #arp = .65*self.state.arp + output*10
        self.state = self.NeuronState(
                         eps0=eps0.detach(),
                         eps1=eps1.detach(),
                         arp=arp.detach(),
                         output = pv.detach())
        self.iter+=1
        return output, pv

class Conv2dDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, im_dims=(28,28), target_size=10, pooling=None, stride=1, dilation=1, padding = 2, alpha=.95, alphas=.9, alpharp =.65, wrp = 0, act = nn.Sigmoid()):
        super(Conv2dDCLLlayer, self).__init__()
        self.im_dims = im_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        if pooling is not None:
            if not hasattr(pooling, '__len__'):
                pooling = (pooling, pooling)

            pool_pad = (pooling[0]-1)//2
            self.pooling = pooling[1]
            print(pooling[0], pooling[1], pool_pad)
            pool_pad = (pooling[1]-1)//2
            self.pooling = pooling[0]
            self.pool = nn.MaxPool2d(kernel_size=pooling[0], stride=pooling[1], padding = pool_pad)
        else:
            self.pooling = 1
            self.pool = lambda x: x
        self.kernel_size = kernel_size
        self.target_size = target_size
        if wrp>0:
            self.i2h = CLLConv2DRRPModule(in_channels,out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, alpha = alpha, alphas = alphas, alpharp = alpharp, wrp = wrp, act = act)
        else:
            self.i2h = CLLConv2DModule(in_channels,out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, alpha = alpha, alphas = alphas, act = act)
        ##best
        #self.i2o = nn.Linear(im_height*im_width*out_channels//self.pooling**2, target_size, bias=False)
        conv_shape = self.i2h.get_output_shape(self.im_dims)
        # actual output shape after pooling a dummy tensor
        self.output_shape = self.pool(torch.zeros(1, *conv_shape)).shape[1:]
        self.i2o = nn.Linear(np.prod(self.output_shape), target_size, bias=True)
        self.i2o.weight.requires_grad = False
        self.i2o.bias.requires_grad = False

    def forward(self, input):
        output, pv = self.i2h(input)
        output, pv = self.pool(output), self.pool(pv)
        flatten = pv.view(pv.shape[0], -1)
        pvoutput = self.i2o(flatten)
        #output = output.detach()
        return output, pvoutput, pv

    def init_hiddens(self, batch_size, init_value = 0):
        self.i2h.init_state(batch_size, self.im_dims, init_value = init_value)
        return self

class DCLLBase(nn.Module):
    def __init__(self, dclllayer, name='DCLLbase', batch_size=48, loss = torch.nn.MSELoss, optimizer = optim.SGD, kwargs_optimizer = {'lr':5e-5}, burnin = 200, collect_stats = False):
        super(DCLLBase, self).__init__()
        self.dclllayer = dclllayer
        self.crit = loss().to(device)
        self.optimizer = optimizer(dclllayer.i2h.parameters(), **kwargs_optimizer)
        self.burnin = burnin
        self.batch_size = batch_size
        self.collect_stats = collect_stats
        self.init(self.batch_size)
        self.name = name

    def init(self, batch_size, init_states = True):
        self.clout = []
        self.mean_activity = []
        self.iter = 0
        if init_states: self.dclllayer.init_hiddens(batch_size, init_value = 0)

    def forward(self, input):
        self.iter+=1
        o, p, pv = self.dclllayer.forward(input)
        if self.collect_stats:
            self.mean_activity.append(o.detach().cpu().numpy().mean())
        return o, p, pv

    def write_stats(self, writer, label, epoch):
        '''
        *writer*: a tensorboard writer
        *label*: label, to append the tensorboard entry
        '''
        if self.collect_stats:
            writer.add_scalar(self.name+'/'+'/meanpv/'+label, np.mean(self.mean_activity), epoch)

    def train(self, input, target):
        output, pvoutput, pv = self.forward(input)
        if self.iter>=self.burnin:
            #self.optimizer.zero_grad()
            self.dclllayer.zero_grad()
            loss = self.crit(pvoutput, target) #+ 4e-6*(torch.norm(pv-.5,2))
            #print(self.crit(pvoutput, target).mean(), 1e-5*((torch.norm(pv-.5,2)).mean()))
            loss.backward()
            self.optimizer.step()
        return output, pvoutput, pv

class DCLLClassification(DCLLBase):
    def forward(self, input):
        o, p, pv = super(DCLLClassification, self).forward(input)
        if self.iter>=self.burnin:
            self.clout.append(p.argmax(1).detach().cpu().numpy())
        return o,p,pv

    def write_stats(self, writer, label, epoch):
        super(DCLLClassification, self).write_stats(writer, label, epoch)
        writer.add_scalar(self.name+'/acc/'+label, self.acc, epoch)

    def accuracy(self, targets):
        cl = np.array(self.clout)
        begin = cl.shape[0]
        self.acc = accuracy_by_vote(cl, targets[-begin:])
        return self.acc

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
#    f = CLLDenseFunction.apply
    pass
