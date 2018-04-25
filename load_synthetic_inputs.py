#!/bin/python
#-----------------------------------------------------------------------------
# File Name : load_synthetic_inputs.py
# Author: Emre Neftci
#
# Creation Date : Wed 18 Apr 2018 10:24:02 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
from npamlib import spiketrains

def input_target_generator(min_duration=400, max_duration=500, Nin=1024, Nout = 10, batch_size = 10):
    """ Generate toy input, target sequences.
    
    """
    assert Nout == batch_size, "for synthetic inputs, batch_size and Nout must be equal"
    T = np.random.randint(min_duration,max_duration,batch_size)
    #inputs = np.array(spiketrains(T = T, N = Nin, rates=[25]*Nin), dtype=np.float32)
    allinputs = np.zeros([batch_size,max_duration,Nin])
    for i in range(batch_size):
        st = spiketrains(T = T[i], N = Nin, rates=np.random.uniform(5,50)).astype(np.float32)
        allinputs[i] =  np.pad(st,((0,max_duration-T[i]),(0,0)),'constant')
    allinputs = np.transpose(allinputs, (1,0,2))

    #Create targets

    alltgt = np.zeros([max_duration, batch_size, Nout], dtype=np.float32)
    for i in range(batch_size):
        t0 = np.linspace(0,T[i],6, dtype='int')[1:]
        t1 = t0+1
        tgt = np.zeros([max_duration,Nout]) #target
        for j in range(len(t0)):
            tgt[t0[j]:t1[j],i] = 1

        alltgt[:,i,i] = tgt[:,i]
    return allinputs, alltgt

def target_convolve(tgt,alpha=8,alphas=5):
    max_duration = tgt.shape[0]
    kernel_alpha = np.exp(-np.linspace(0,10*alpha,dtype='float')/alpha)
    kernel_alpha /= kernel_alpha.sum()
    kernel_alphas = np.exp(-np.linspace(0,10*alphas,dtype='float')/alphas)
    kernel_alphas /= kernel_alphas.sum()
    tgt = tgt.copy()
    for i in range(tgt.shape[1]):
        for j in range(tgt.shape[2]):
            tmp=np.convolve(np.convolve(tgt[:,i,j],kernel_alpha),kernel_alphas)[:max_duration]
            tgt[:,i,j] = tmp
    return tgt/tgt.max()
