#!/bin/python
#-----------------------------------------------------------------------------
# File Name : tensorflow_snn.py
# Author: Emre Neftci
#
# Creation Date : Fri 06 Apr 2018 03:49:58 PM PDT
# Last Modified : Mon 23 Apr 2018 10:00:50 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# modified from https://rdipietro.github.io/tensorflow-scan-examples/#defining-the-rnn-model-from-scratch
#----------------------------------------------------------------------------- 


import matplotlib.pyplot as plt

import os
import shutil
import numpy as np
import tensorflow as tf
from npamlib import spiketrains

from tensorflow.python.ops import functional_ops

from npamlib import plotLIF
from load_synthetic_inputs import *
from load_dvs_gestures import *
from libdcll import *


Nfeat1=16
Nfeat2=32
Nfeat3=48
Nfeat4=64
Nfeat5=64

Nin = [32,32,1]
Nhid1 = [32,32,Nfeat1]
Nhid2in = [16,16,Nfeat1]
Nhid2 = [16,16,Nfeat2]
Nhid3in = [8,8,Nfeat2]
Nhid3 = [8,8,Nfeat3]
Nout = 11
T = 500
nepochs = 5000
max_target = .9
batch_size = 64


max_layers = 4
layers = [None for i in range(max_layers)]
states = [None for i in range(max_layers)]

layers[0], states[0], output0 = DCNNConvLayer(feat_out=Nfeat1,
        ksize=7,
        input_shape=[32,32,1]     ,
        target_size = Nout,
        layer_input = None   ,
        batch_size=batch_size,
        tau = 10,
        taus = 20,
        pooling=2)

layers[1], states[1], output1 = DCNNConvLayer(feat_out=Nfeat2,
        ksize=7,
        input_shape=[16,16,Nfeat1],
        target_size = Nout,
        layer_input = output0,
        batch_size = batch_size,
        tau = 10,
        taus = 20,
        pooling=2)

layers[2], states[2], output2 = DCNNConvLayer(feat_out=Nfeat3,
        ksize=7,
        input_shape=[8, 8,Nfeat2],
        target_size = Nout,
        layer_input = output1,
        batch_size=batch_size,
        tau = 10,
        taus = 20,
        pooling=1)

layers[3], states[3], output3 = DCNNConvLayer(feat_out=Nfeat4,
        ksize=7,
        input_shape=[8, 8,Nfeat3],
        target_size = Nout,
        layer_input = output2,
        batch_size=batch_size,
        tau = 10,
        taus = 20,
        pooling=2)



if __name__ == '__main__':
    preds = [states[i][5] for i in range(len(states))]
    train_W_ops = [states[i][-1][-1] for i in range(len(states))]
    train_b_ops = [states[i][-2][-1] for i in range(len(states))]
    train_ops = tf.group(*(train_W_ops + train_b_ops))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    gen_train, gen_test = create_data(batch_size=batch_size, chunk_size = T)

    acc_train = []
    acc_test = []
    lr = 10e-5

    gen_inputs_test, gen_targets_test = next(gen_test)
    for i in range(nepochs):
        gen_inputs, gen_targets = next(gen_train)
        inputs = gen_inputs.swapaxes(0,1)
        targets_original = gen_targets.copy()
        targets = [None]*len(states)
        for j in range(len(states)):
            targets[j] = targets_original #target_convolve(targets_original ,alpha=layers[j].tau,alphas=layers[j].taus)*max_target
            targets_original = targets[j].copy()

        feed_dict = {layers[0].inputs : inputs}
        feed_dict.update({layers[k].targets : targets[k] for k in range(len(states))})
        feed_dict.update({layers[k].mod_lr:lr for k in range(len(states))})
        #train epoch

        ps, _ = sess.run([preds,train_ops], feed_dict)
        accs = [np.mean(p[100:].cumsum(axis=0).argmax(axis=2)==targets_original[100:].cumsum(axis=0).argmax(axis=2)) for p in ps]
        acc_train.append([i]+accs)
        print(' Train '.join('{:1.3f}'.format(k) for k in acc_train[-1]))
        if (i%2)==0:
            inputs = gen_inputs_test.swapaxes(0,1)
            targets_original = gen_targets_test.copy()
            targets = [None]*len(states)
            for j in range(len(states)):
                targets[j] = targets_original #target_convolve(targets_original ,alpha=layers[j].tau,alphas=layers[j].taus)*max_target
                targets_original = targets[j].copy()
            ##test epoch
            feed_dict = {layers[0].inputs : inputs}
            feed_dict.update({layers[k].targets : targets[k] for k in range(len(states))})
            feed_dict.update({layers[k].mod_lr:0. for k in range(len(states))})  
            ps = sess.run(preds,feed_dict)
            accs = [np.mean(p[100:].cumsum(axis=0).argmax(axis=2)==targets_original[100:].cumsum(axis=0).argmax(axis=2)) for p in ps]
            acc_test.append([i]+accs)
            print(' Test  '.join('{:1.3f}'.format(k) for k in acc_test[-1]))

