#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : Mon 12 Nov 2018 03:11:40 PM PST
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
soft_threshold = torch.sigmoid

device = 'cuda'

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

def accuracy_by_mse(pvoutput, labels):
    return torch.sum((pvoutput - labels)**2).item()

class CLLDenseModule(nn.Module):
    NeuronState = namedtuple('NeuronState', ['eps0', 'eps1'])
    def __init__(self, in_channels, out_channels, bias=True, alpha = .9, alphas=.85, act = nn.Sigmoid(), spiking = True, random_tau = False):
        super(CLLDenseModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.random_tau = random_tau
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]), requires_grad = False)
        self.tau_m__dt = torch.nn.Parameter(torch.Tensor([1./(1-self.alpha)]), requires_grad = False)
        self.alphas = torch.nn.Parameter(torch.Tensor([alphas]), requires_grad = False)
        self.tau_s__dt = torch.nn.Parameter(torch.Tensor([1./(1-self.alphas)]), requires_grad = False)
        self.spiking = spiking

        if spiking:
            self.output_act = lambda x: (x>0).float()
        else:
            self.output_act = lambda x: x
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv*1e-2, stdv*1e-2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_state(self, batch_size, init_value = 0):
        self.state = self.NeuronState(
            eps0 = torch.zeros(batch_size, self.in_channels ).detach().to(device) + init_value,
            eps1 = torch.zeros(batch_size, self.in_channels ).detach().to(device) + init_value
            )
        if self.random_tau:
            self.randomize_tau()
        return self.state
    
    def randomize_tau(self, low=[5,5], high=[10,35]):
        taum = np.random.uniform(low[1], high[1], size = [self.in_channels])*1e-3
        taus = np.random.uniform(low[0], high[0], size = [self.in_channels])*1e-3
        self.alpha = torch.nn.Parameter(torch.Tensor(1-1e-3/taum).to(device), requires_grad = False)
        self.tau_m__dt = torch.nn.Parameter(1./(1-self.alpha), requires_grad = False) 
        self.alphas = torch.nn.Parameter(torch.Tensor(1-1e-3/taus).to(device), requires_grad = False)
        self.tau_s__dt = torch.nn.Parameter(1./(1-self.alphas), requires_grad = False)


    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                           .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0])

        eps0 = input*self.tau_s__dt + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0*self.tau_m__dt
        vmem = F.linear(eps1, self.weight, self.bias)
        pv = self.act(vmem)
        output = self.output_act(vmem)
        # update the neuronal state
        self.state = self.NeuronState(eps0=eps0.detach(),
                                      eps1=eps1.detach())

        return output, pv, vmem

class CLLDenseRRPModule(CLLDenseModule):
    NeuronState = namedtuple('NeuronState', ('eps0', 'eps1', 'arp'))
    
    def __init__(self, in_channels, out_channels, bias=True, alpha = .95, alphas=.9, alpharp = .65, wrp = 100, act = nn.Sigmoid(), spiking = True, random_tau = False):
        super(CLLDenseRRPModule, self).__init__(in_channels, out_channels, bias, alpha, alphas, act, spiking = spiking, random_tau = random_tau)
        self.wrp=wrp
        self.alpharp=alpharp

    def init_state(self, batch_size, init_value = 0):
        self.state = self.NeuronState(
            eps0 = torch.zeros(batch_size, self.in_channels ).to(device) + init_value,
            eps1 = torch.zeros(batch_size, self.in_channels ).to(device) + init_value,
            arp = torch.zeros(batch_size, self.out_channels).to(device) + init_value,
        )
        return self.state
    

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0])

        eps0 = input*self.tau_s__dt + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0*self.tau_m__dt
        pvmem = F.linear(eps1, self.weight, self.bias)
        arp     = self.alpharp*self.state.arp 
        outpvmem = pvmem+arp
        output = (outpvmem>0).float()
        pv = self.act(outpvmem)
        #pv = self.act(outpvmem)
        if not self.spiking: raise Exception('Refractory not allowed in non-spiking mode')
        arp -= output*self.wrp
        self.state = self.NeuronState(
                         eps0=eps0.detach(),
                         eps1=eps1.detach(),
                         arp=arp.detach())

        return output, pv, outpvmem

class DenseDCLLlayer(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            target_size=None,
            bias= True,
            alpha=.9,
            alphas = .85,
            alpharp =.65,
            wrp = 0.,
            act = nn.Sigmoid(),
            lc_dropout=False,
            lc_ampl=.5,
            spiking = True,
            random_tau = False,
            output_layer = False):

        if (target_size is None):
            target_size = out_channels
        super(DenseDCLLlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lc_ampl = lc_ampl
        self.target_size = target_size
        self.output_layer = False
        if wrp>0:
            self.i2h = CLLDenseRRPModule(in_channels,out_channels, alpha = alpha, alphas = alphas, alpharp = alpharp, wrp = wrp, bias = bias, act = act, spiking = spiking, random_tau = random_tau)
        else:
            self.i2h = CLLDenseModule(in_channels,out_channels, alpha=alpha, alphas=alphas, bias = bias, act = act, spiking = spiking, random_tau = random_tau)
        self.i2o = nn.Linear(out_channels, target_size, bias=bias)
        self.i2o.weight.requires_grad = False
        if bias:
            self.i2o.bias.requires_grad = False
        # self.softmax = nn.LogSoftmax(dim=1)
        self.input_size = self.out_channels
        self.reset_lc_parameters()
        self.lc_dropout = lc_dropout
        self.lc_ampl = lc_ampl

        if lc_dropout is not False:
            self.dropout = torch.nn.Dropout(p=lc_dropout)
        else:
            self.dropout = lambda x: x

    def reset_lc_parameters(self):
        stdv = self.lc_ampl / math.sqrt(self.i2o.weight.size(1))
        self.i2o.weight.data.uniform_(-stdv, stdv)
        if self.i2o.bias is not None:
            self.i2o.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input   = input.view(-1,self.in_channels)
        output, pv, pvmem = self.i2h(input)
        pvoutput = self.dropout(self.i2o(pv))
        output = output.detach()
        return output, pvoutput, pv, pvmem

    def init_hiddens(self, batch_size, init_value = 0):
        self.i2h.init_state(batch_size, init_value = init_value)
        return self

    def reset_tracks(self, mask = None):
        if mask is None:
            self.init_hiddens(self.i2h.state.isyn.shape[0])
            return
        for field in self.i2h.state:
            field[mask] = 0.

class DenseDCLLlayerDiscrete(DenseDCLLlayer):
    def forward(self, input):
        input   = input.view(-1,self.in_channels)
        output, pv, pvmem = self.i2h(input)
        pvoutput = self.dropout(soft_threshold(self.i2o(pv)))                
        return output, pvoutput, pv, pvmem

class AnalogDenseDCLLlayer(nn.Module):
    def __init__(self, in_channels, out_channels, target_size, act = nn.Sigmoid()):
        super(AnalogDenseDCLLlayer, self).__init__()
        self.i2h = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )
        self.i2o = nn.Linear(out_channels, target_size)
        # Disable gradients on weights and biases at the local error layer
        self.i2o.weight.requires_grad = False
        self.i2o.bias.requires_grad = False

    def forward(self, x):
        out = self.i2h(x)       # Hidden layer activity
        pout = self.i2o(out)    # Error layer acticity
        out = out.detach()      # Disable learning on the output
        return out, pout

class ContinuousConv2D(nn.Module):
    NeuronState = namedtuple('NeuronState', ('eps0', 'eps1'))
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            alpha = .95,
            alphas=.9,
            act = nn.Sigmoid(),
            random_tau = False,
            spiking = True,
            **kwargs):
        super(ContinuousConv2D, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
    
        self.random_tau = random_tau
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.act = act
        self.spiking = spiking
        if spiking:
            self.output_act = lambda x: (x>0).float()
        else:
            self.output_act = lambda x: x

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.alpha = torch.nn.Parameter(torch.Tensor([alpha]), requires_grad = False)
        self.tau_m__dt = torch.nn.Parameter(torch.Tensor([1./(1-self.alpha)]), requires_grad = False)
        self.alphas = torch.nn.Parameter(torch.Tensor([alphas]), requires_grad = False)
        self.tau_s__dt = torch.nn.Parameter(torch.Tensor([1./(1-self.alphas)]), requires_grad = False)
        print(act)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n) / 250
        self.weight.data.uniform_(-stdv*1e-2, stdv*1e-2)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_output_shape(self, im_dims):
        im_height = im_dims[0]
        im_width = im_dims[1]
        height = ((im_height+2*self.padding-self.dilation*(self.kernel_size[0]-1)-1)//self.stride+1)
        weight = ((im_width+2*self.padding-self.dilation*(self.kernel_size[1]-1)-1)//self.stride+1)
        return height,weight

    def init_state(self, batch_size, im_dims, init_value = 0):
        input_shape = [batch_size, self.in_channels, im_dims[0], im_dims[1]]
        isyn_shape =  torch.Size([batch_size]) + self.get_output_shape(im_dims)

        self.state = self.NeuronState(
            eps0 = torch.zeros(input_shape).detach().to(device)+init_value,
            eps1 = torch.zeros(input_shape).detach().to(device)+init_value
            )
        
        if self.random_tau:
            self.randomize_tau(im_dims)
            self.random_tau=False

        return self.state
    
    def randomize_tau(self, im_dims, low=[5,5], high=[10,35]):
        taum = np.random.uniform(low[1], high[1], size = [self.in_channels])*1e-3
        taus = np.random.uniform(low[0], high[0], size = [self.in_channels])*1e-3
        taum = np.broadcast_to(taum, (im_dims[0], im_dims[1], self.in_channels)).transpose(2,1,0)
        taus = np.broadcast_to(taus, (im_dims[0], im_dims[1], self.in_channels)).transpose(2,1,0)
        self.alpha = torch.nn.Parameter(torch.Tensor(1-1e-3/taum).to(device), requires_grad = False)
        self.tau_m__dt = torch.nn.Parameter(1./(1-self.alpha), requires_grad = False) 
        self.alphas = torch.nn.Parameter(torch.Tensor(1-1e-3/taus).to(device), requires_grad = False)
        self.tau_s__dt = torch.nn.Parameter(1./(1-self.alphas), requires_grad = False)

    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logging.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0], input.shape[2:4])

       # isyn = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
       # isyn += self.alphas*self.state.isyn
       # vmem = self.alpha*self.state.vmem + isyn
        eps0 = input*self.tau_s__dt + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0*self.tau_m__dt
        pvmem = F.conv2d(eps1, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        pv = self.act(pvmem)
        output = self.output_act(pvmem)


        ##best
        #arp = .65*self.state.arp + output*10
        self.state = self.NeuronState( eps0=eps0.detach(),
                                       eps1=eps1.detach())
        return output, pv, pvmem

    def init_prev(self, batch_size, im_dims):
        return torch.zeros(batch_size, self.in_channels, im_dims[0], im_dims[1])

class ContinuousRelativeRefractoryConv2D(ContinuousConv2D):
    NeuronState = namedtuple('NeuronState', ('eps0', 'eps1', 'arp'))

    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=2,
            dilation=1,
            groups=1,
            bias=True,
            alpha = .95,
            alphas=.9,
            alpharp = .65,
            wrp = 1,
            act = nn.Sigmoid(), 
            random_tau=False,
            **kwargs):
        '''
        Continuous local learning with relative refractory period. No isyn or vmem dynamics for speed and memory.
        *wrp*: weight for the relative refractory period
        '''
        super(ContinuousRelativeRefractoryConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, alpha, alphas, act)

        print("Relative RP")
        ##best
        #self.tarp=10
        self.wrp=wrp
        self.alpharp=alpharp
        self.tau_rp__dt = 1./(1-self.alpharp)
        self.iter=0
        self.tau_set=False
        self.random_tau = random_tau

    def init_state(self, batch_size, im_dims, init_value = 0):
        input_shape = [batch_size, self.in_channels, im_dims[0], im_dims[1]]
        output_shape =  torch.Size([batch_size, self.out_channels]) + self.get_output_shape(im_dims)

        self.state = self.NeuronState(
            eps0 = torch.zeros(input_shape).to(device)+init_value,
            eps1 = torch.zeros(input_shape).to(device)+init_value,
            arp = torch.zeros(output_shape).to(device),
            )

        if self.random_tau:
            self.randomize_tau(im_dims)
            self.random_tau=True

        return self.state
    
    def forward(self, input):
        # input: input tensor of shape (minibatch x in_channels x iH x iW)
        # weight: filters of shape (out_channels x (in_channels / groups) x kH x kW)
        if not (input.shape[0] == self.state.eps0.shape[0] == self.state.eps1.shape[0]):
            logger.warning("Batch size changed from {} to {} since last iteration. Reallocating states."
                            .format(self.state.eps0.shape[0], input.shape[0]))
            self.init_state(input.shape[0], input.shape[2:4])

#         eps0 = input + self.alphas * self.state.eps0
#         eps1 = self.alpha * self.state.eps1 + eps0
        eps0 = input*self.tau_s__dt + self.alphas * self.state.eps0
        eps1 = self.alpha * self.state.eps1 + eps0*self.tau_m__dt
        pvmem = F.conv2d(eps1, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        arp     = self.alpharp*self.state.arp 
        outpvmem = pvmem+arp
        output = (outpvmem>0).float()
        pv = self.act(outpvmem)
        #pv = self.act(outpvmem)
        if not self.spiking: raise Exception('Refractory not allowed in non-spiking mode')
        arp -= output*self.wrp
        self.state = self.NeuronState(
                         eps0=eps0.detach(),
                         eps1=eps1.detach(),
                         arp=arp.detach())

        return output, pv, outpvmem

class Conv2dDCLLlayer(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size = 5,
            im_dims = (28,28),
            target_size = 10,
            pooling = None,
            stride = 1,
            dilation = 1,
            padding = 2,
            alpha=.95,
            alphas=.9,
            alpharp =.65,
            wrp = 0,
            act = nn.Sigmoid(),
            lc_dropout = False,
            lc_ampl = .5,
            spiking = True,
            random_tau = False,
            output_layer = False):

        super(Conv2dDCLLlayer, self).__init__()
        self.im_dims = im_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lc_ampl = lc_ampl
        self.output_layer = output_layer

        #The following code builds the pooling into the module
        if pooling is not None:
            if not hasattr(pooling, '__len__'):  pooling = (pooling, pooling)
            pool_pad = (pooling[0]-1)//2
            self.pooling = pooling[1]
            pool_pad = (pooling[1]-1)//2
            self.pooling = pooling[0]
            self.pool = nn.MaxPool2d(kernel_size=pooling[0], stride=pooling[1], padding = pool_pad)
        else:
            self.pooling = 1
            self.pool = lambda x: x
        self.kernel_size = kernel_size
        self.target_size = target_size
        if wrp>0:
            if not spiking: raise Exception('Non-spiking not allowed with refractory neurons')
            self.i2h = ContinuousRelativeRefractoryConv2D(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, alpha = alpha, alphas = alphas, alpharp = alpharp, wrp = wrp, act = act, random_tau=random_tau)
        else:
            self.i2h = ContinuousConv2D(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, alpha = alpha, alphas = alphas, act = act, spiking= spiking, random_tau = random_tau)
        conv_shape = self.i2h.get_output_shape(self.im_dims)
        print('Conv2D Layer ', self.im_dims, conv_shape, self.in_channels, self.out_channels, kernel_size, dilation, padding, stride)
        self.output_shape = self.pool(torch.zeros(1, *conv_shape)).shape[1:]
        self.i2o = nn.Linear(np.prod(self.get_flat_size()), target_size, bias=True)
        self.i2o.weight.requires_grad = False
        if lc_dropout is not False:
            self.dropout = torch.nn.Dropout(p=lc_dropout)
        else:
            self.dropout = lambda x: x
        self.i2o.bias.requires_grad = False

        if output_layer:
            self.output_ = nn.Linear(np.prod(self.get_flat_size()), target_size, bias=True)

        self.reset_lc_parameters()

    def reset_lc_parameters(self):
        stdv = self.lc_ampl / math.sqrt(self.i2o.weight.size(1))
        self.i2o.weight.data.uniform_(-stdv, stdv)
        if self.i2o.bias is not None:
            self.i2o.bias.data.uniform_(-stdv, stdv)

    def get_flat_size(self):
        w,h = self.get_output_shape()
        return int(w*h*self.out_channels)

    def get_output_shape(self):
        conv_shape = self.i2h.get_output_shape(self.im_dims)
        height = conv_shape[0]//self.pooling
        weight = conv_shape[1]//self.pooling
        return height,weight

    def forward(self, input):
        output, pv, pvmem = self.i2h(input)
        output, pv = self.pool(output), self.pool(pv)
        flatten = pv.view(pv.shape[0], -1)
        pvoutput = self.dropout(self.i2o(flatten))
        
        if self.output_layer:
            custom_output = self.output_(flatten.detach())
        else:
            custom_output = output
            
        return custom_output, pvoutput, pv, pvmem

    def init_hiddens(self, batch_size, init_value = 0):
        self.i2h.init_state(batch_size, self.im_dims, init_value = init_value)
        return self


#class Conv2dDCLLlayerDiscrete(Conv2dDCLLlayer):
#    def forward(self, input):
#        output, pv, pvmem = self.i2h(input)
#        output, pv = self.pool(output), self.pool(pv)
#        flatten = pv.view(pv.shape[0], -1)
#        pvoutput = self.dropout(soft_threshold(self.i2o(flatten)))
#        #output = output.detach()
#        return output, pvoutput, pv, self.pool(pvmem)

class DCLLBase(nn.Module):
    def __init__(self, dclllayer, name='DCLLbase', batch_size=48, loss = torch.nn.MSELoss, optimizer = optim.SGD, kwargs_optimizer = {'lr':5e-5}, burnin = 200, collect_stats = False):
        '''
        *dclllayer*: layer that supports local learning
        *batch_size*: Used for initialization
        *loss*: torch loss class
        *optimizer* : torch optimizer class
        *kwargs_optimizer*: options to be passed to optimizer
        *collect_stats*: boolean whether statistics (weight bias histograms) under write_stats should be collected during learning)
        '''
        super(DCLLBase, self).__init__()
        self.dclllayer = dclllayer
        self.crit = loss().to(device)
        self.output_crit = nn.CrossEntropyLoss().to(device)
        self.optimizer = optimizer(dclllayer.i2h.parameters(), **kwargs_optimizer)
        if self.dclllayer.output_layer:
            self.optimizer2 = optimizer(dclllayer.output_.parameters(), lr = 1e-4)
        self.burnin = burnin
        self.batch_size = batch_size
        self.collect_stats = collect_stats
        self.init(self.batch_size)
        self.stats_bins = np.linspace(0,1,20)
        self.name = name

    def init(self, batch_size, init_states = True):
        self.clout = []
        self.activity_hist = []
        self.iter = 0
        if init_states: self.dclllayer.init_hiddens(batch_size, init_value = 0)

    def forward(self, input):
        self.iter+=1
        o, p, pv, pvmem = self.dclllayer.forward(input)
        if self.collect_stats:
            if (self.iter%20)==0:
                self.activity_hist.append(np.histogram(pv.detach().cpu().numpy(), bins = self.stats_bins)[0])
        return o, p, pv, pvmem

    def write_stats(self, writer, label, epoch):
        '''
        *writer*: a tensorboard writer
        *label*: label, to append the tensorboard entry
        '''
        writer.add_histogram(self.name+'/weight',
                             self.dclllayer.i2h.weight.flatten(),
                             epoch)
        writer.add_histogram(self.name+'/bias',
                             self.dclllayer.i2h.bias.flatten(),
                             epoch)
        if self.collect_stats:
            pd = np.mean(self.activity_hist,axis=0)
            pd = pd /pd.sum()

            name = self.name+'/'+'/low_pv/'+label

            writer.add_scalar(name, (pd[0]), epoch)

            name = self.name+'/'+'/high_pv/'+label

            writer.add_scalar(name, (pd[-1]), epoch)

            print(self.name + " low:{0:1.3} high:{1:1.3}".format(pd[0],pd[-1]))


    def train_dcll(self, input, target, do_train = True, regularize = 0.05):
        output, pvoutput, pv, pvmem = self.forward(input)
        if self.iter>=self.burnin:
            self.dclllayer.zero_grad()
            tgt_loss = self.crit(pvoutput, target) 
            if self.dclllayer.output_layer:
                out_loss = self.output_crit(output, target.argmax(-1)) 
                tgt_loss += out_loss
            if regularize>0:
                reg_loss = 200e-1*regularize*torch.mean(torch.relu(pvmem+.01))
                reg2_loss = 1e-1*regularize*(torch.relu(.1-torch.mean(pv)))
                loss = tgt_loss + reg_loss + reg2_loss
            else:
                loss = tgt_loss
            loss.backward()
            if do_train:
                self.optimizer.step()
                if self.dclllayer.output_layer: self.optimizer2.step()
        else:
            tgt_loss = torch.Tensor([0])

        return output, pvoutput, pv, pvmem, tgt_loss.detach()

class DCLLClassification(DCLLBase):
    def forward(self, input):
        o, p, pv, pvmem = super(DCLLClassification, self).forward(input)        
        if self.iter>=self.burnin:
            if self.dclllayer.output_layer:
                self.clout.append(o.argmax(1).detach().cpu().numpy())
            else:
                self.clout.append(p.argmax(1).detach().cpu().numpy())
        return o,p,pv, pvmem

    def write_stats(self, writer, label, epoch):
        super(DCLLClassification, self).write_stats(writer, label, epoch)
        writer.add_scalar(self.name+'/acc/'+label, self.acc, epoch)

    def accuracy(self, targets):
        cl = np.array(self.clout)
        begin = cl.shape[0]
        self.acc = accuracy_by_vote(cl, targets[-begin:])
        return self.acc

class DCLLRegression(DCLLBase):
    def forward(self, input):
        o, p, pv, pvmem = super(DCLLRegression, self).forward(input)
        if self.iter>=self.burnin:
            self.clout.append(p)
        return o,p,pv, pvmem

    def write_stats(self, writer, label, epoch):
        super(DCLLRegression, self).write_stats(writer, label, epoch)
        writer.add_scalar(self.name+'/acc/'+label, self.acc, epoch)

    def accuracy(self, targets):
        cl = torch.stack(self.clout, dim=0)
        begin = cl.shape[0]
        self.acc = accuracy_by_mse(cl, targets[-begin:])
        return self.acc

class DCLLGeneration(DCLLBase):

    def init(self, batch_size, init_states = True):
        self.vmem_out = []
        self.spikes_out = []
        self.clout = []
        self.iter = 0
        if init_states: self.dclllayer.init_hiddens(batch_size, init_value = 0)

    def forward(self, input):
        o, p, pv, pvmem = super(DCLLGeneration, self).forward(input)
        self.clout.append(p.detach().cpu().numpy())
        self.spikes_out.append(o.detach().cpu().numpy())
        self.vmem_out.append(self.dclllayer.i2h.state[1].detach().cpu().numpy())
        return o,p, pv, pvmem

def save_dcllslices(directory,slices):
    for i,s in enumerate(slices):
        torch.save(s.state_dict(), directory+'/slice_state{0}.pkl'.format(i))

def load_dcllslices(directory,slices):
    for i,s in enumerate(slices):
        s.load_state_dict(torch.load(directory+'/slice_state{0}.pkl'.format(i)))



if __name__ == '__main__':
    #Test dense gradient
#    f = CLLDenseFunction.apply
    pass
