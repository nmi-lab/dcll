#!/bin/python
#-----------------------------------------------------------------------------
# File Name : tensorflow_snn.py
# Author: Emre Neftci
#
# Creation Date : Fri 06 Apr 2018 03:49:58 PM PDT
# Last Modified : Mon 23 Apr 2018 09:59:32 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# modified from https://rdipietro.github.io/tensorflow-scan-examples/#defining-the-rnn-model-from-scratch
#----------------------------------------------------------------------------- 
from __future__ import division, print_function
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import tensorflow as tf
from npamlib import spiketrains
from tensorflow.python.ops import functional_ops
from npamlib import plotLIF
from load_mnist import *
from libdcll import *

T = 500
nepochs = 5000
max_target = 1.0
Nin = [28,28,1]
Nout = 10
batch_size = 64

def build_mlp():
    Nfeat1=500
    Nfeat2=500
    Nfeat3=500

    Nhid1 = [Nfeat1]
    Nhid2 = [Nfeat2]
    Nhid3 = [Nfeat3]


    max_layers = 3
    layers = [None for i in range(max_layers)]
    states = [None for i in range(max_layers)]

    Nin_ = [np.prod(Nin)]
    layers[0], states[0], output0 = DCNNDenseLayer(feat_out=Nfeat1, input_shape = Nin_     , layer_input = None   , batch_size=batch_size, target_size = Nout, tau=15, taus=20)
    layers[1], states[1], output1 = DCNNDenseLayer(feat_out=Nfeat2, input_shape = [Nfeat1], layer_input = output0, batch_size=batch_size, target_size = Nout, tau=15, taus=20)
    layers[2], states[2], output2 = DCNNDenseLayer(feat_out=Nfeat3, input_shape = [Nfeat2], layer_input = output1, batch_size=batch_size, target_size = Nout, tau=15, taus=20)
    return layers,states

def build_cnn():
    Nfeat1=64
    Nfeat2=96
    Nfeat3=128
    tau = 15
    taus = 25

    Nhid1 = [28,28,Nfeat1]
    Nhid2in = [14,14,Nfeat1]
    Nhid2 = [14,14,Nfeat2]
    Nhid3in = [7,7,Nfeat2]
    Nhid3 = [7,7,Nfeat3]

    max_layers = 3
    layers = [None for i in range(max_layers)]
    states = [None for i in range(max_layers)]

    layers[0], states[0], output0 = DCNNConvLayer(
            feat_out=Nfeat1,
            ksize=5,
            input_shape=Nin     ,
            target_size = Nout,
            layer_input = None   ,
            batch_size=batch_size,
            pooling=2,
            tau = tau,
            taus = taus)

    layers[1], states[1], output1 = DCNNConvLayer(
            feat_out=Nfeat2,
            ksize=5,
            input_shape=Nhid2in,
            target_size = Nout,
            layer_input = output0,
            batch_size=batch_size,
            pooling=2, 
            tau = tau,
            taus = taus)

    layers[2], states[2], output2 = DCNNConvLayer(
            feat_out=Nfeat3,
            ksize=5,
            input_shape=Nhid3in,
            target_size = Nout,
            layer_input = output1,
            batch_size=batch_size,
            pooling=1,
            tau = tau,
            taus = taus)
    return layers,states

layers,states = build_cnn()

if __name__ == '__main__':

    preds = [states[i][5] for i in range(len(states))]
    train_W_ops = [states[i][-1][-1] for i in range(len(states))]
    train_b_ops = [states[i][-2][-1] for i in range(len(states))]
    train_ops = tf.group(*(train_W_ops + train_b_ops))

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())

    gen_train, gen_valid, gen_test = create_data(batch_size=batch_size)

    acc_train = []
    acc_test = []
    lr = 1e-4

    for i in range(nepochs):
        gen_inputs, gen_targets = image2spiketrain(*gen_train.next())
        inputs = gen_inputs
        targets_original = gen_targets
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
        print(' '.join('{:1.3f}'.format(k) for k in acc_train[-1]))
        if (i%20)==0:
            gen_inputs, gen_targets = image2spiketrain(*gen_test.next())
            inputs = gen_inputs
            targets_original = gen_targets
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
            print(' '.join('{:1.3f}'.format(k) for k in acc_test[-1]))

