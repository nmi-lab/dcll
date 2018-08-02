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
import pyNSATlib as nsat
import time
# from pyNSATlib import ConfigurationNSAT, exportAER, build_SpikeList


# Explicit tensorflow config gives more control
gpuid = 0 # An index of which gpu to use. 
os.environ['KERAS_BACKEND'] = 'tensorflow'
# Comment the line below and uncomment the line below that to switch from GPU to CPU only execution
os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(gpuid) # (Empty) List of gpu indices that TF can see.
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
CONFIG = tf.ConfigProto(device_count = {'GPU': 1}, log_device_placement=False, allow_soft_placement=False) # Only use a single GPU.
# Uncomment the two lines below to allow dynamic memory growth rather than having cuda use all available memory from the start
#CONFIG.gpu_options.allocator_type = 'BFC'
#CONFIG.gpu_options.allow_growth = True # Prevents tf from grabbing all gpu memory.
sess = tf.Session(config=CONFIG)
from keras import backend as K
K.set_session(sess)

def setup_mnist():
    global T, nepochs, max_target, Nin, Nout, batch_size, layers, states
    layers,states = build_cnn()
    T = 1000
    nepochs = 100
    max_target = 1.0
    Nin = [28,28,1]
    Nout = 10
    batch_size = 32 #Ensure that batchsize can divide 50000 and 10000

def setup_nsat():
    print('Begin %s:setup_nsat()' % (os.path.splitext(os.path.basename(__file__))[0]))
    sim_ticks = 5000        # Simulation time
    N_CORES = 1             # Number of cores
    N_NEURONS = [2]         # Number of neurons per core (list)
    N_INPUTS = [0]          # Number of inputs per core (list)
    N_STATES = [4]          # Number of states per core (list)
#     global cfg

    # Constants
    XMAX = nsat.XMAX
    XMIN = nsat.XMIN
    OFF = -16
    MAX = nsat.MAX
    MIN = nsat.XMIN

    # Main class instance
    cfg = nsat.ConfigurationNSAT(sim_ticks=sim_ticks,
                                 N_CORES=N_CORES,
                                 N_INPUTS=N_INPUTS,
                                 N_NEURONS=N_NEURONS,
                                 N_STATES=N_STATES,
                                 monitor_states=True,
                                 ben_clock=True)

    # Transition matrix A
    cfg.core_cfgs[0].A[0] = [[-6,  OFF, OFF, OFF],
                             [0, -11, OFF, OFF],
                             [0, OFF, -8, OFF],
                             [OFF, OFF, OFF, OFF]]

    # Sign matrix sA
    cfg.core_cfgs[0].sA[0] = [[-1, 1, 1, 1],
                              [-1, -1, 1, 1],
                              [1, 1, -1, 1],
                              [1, 1, 1, -1]]

    # Bias
    cfg.core_cfgs[0].b[0] = np.array([600, 0, 0, 1], dtype='int')
    # Threshold
    cfg.core_cfgs[0].Xth[0] = XMAX
    # Reset value
    cfg.core_cfgs[0].Xreset[0] = np.array([0, XMAX, XMAX, XMAX], 'int')
    # Turn reset on
    cfg.core_cfgs[0].XresetOn[0] = np.array([True, False, False, False],
                                            'bool')
    # Global modulator state (e.g. Dopamine)
    cfg.core_cfgs[0].modstate[0] = 3

    # Synaptic weights
    N_UNITS = N_INPUTS[0] + N_NEURONS[0]        # Total number of units
    W = np.zeros([N_UNITS, N_UNITS, N_STATES[0]], 'int')
    W[0, 0, 1] = 115
    W[0, 1, 2] = 125
    W[1, 1, 1] = 115
    W[1, 0, 2] = 125
    cfg.core_cfgs[0].W = W

    # Adjacent matrix
    CW = np.zeros(W.shape, dtype='int')
    CW[0, 0, 1] = 1
    CW[0, 1, 2] = 1
    CW[1, 1, 1] = 1
    CW[1, 0, 2] = 1
    cfg.core_cfgs[0].CW = CW

    # Mapping between neurons and NSAT parameters groups
    cfg.core_cfgs[0].nmap = np.zeros((N_NEURONS[0],), dtype='int')

    # Write C NSAT parameters binary files
    c_nsat_writer = nsat.C_NSATWriter(cfg, path='/tmp', prefix='test_adapting')
    c_nsat_writer.write()

    # Write Intel FPGA parameters hex files
#    intel_fpga_writer = nsat.IntelFPGAWriter(cfg, path='.',
#                                             prefix='test_adapting')
#    intel_fpga_writer.write()
#    intel_fpga_writer.write_globals()
    print('End %s:setup_nsat()' % (os.path.splitext(os.path.basename(__file__))[0]))
    return c_nsat_writer.fname


def run_nsat(fnames):
    # Call the C NSAT
    print('Begin %s:run_nsat()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    cfg = nsat.ConfigurationNSAT.readfileb(fnames.pickled)
    nsat.run_c_nsat(fnames)

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, fnames)
    states = c_nsat_reader.read_c_nsat_states()
    time_core0, states_core0 = states[0][0], states[0][1]

    # Plot the results
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 5):
        ax = fig.add_subplot(4, 1, i)
        ax.plot(states_core0[:-1, 0, i-1], 'b', lw=3)
        
    plt.savefig('/tmp/%s.png' % (os.path.splitext(os.path.basename(__file__))[0]))
    plt.close()
    print('End %s:run_nsat()' % (os.path.splitext(os.path.basename(__file__))[0]))


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


def mnist_benchmark():
    print('Begin %s:mnist_benchmark()' % (os.path.splitext(os.path.basename(__file__))[0]))
    
    start_t = time.perf_counter()

    preds = [states[i][5] for i in range(len(states))]
    train_W_ops = [states[i][-1][-1] for i in range(len(states))]
    train_b_ops = [states[i][-2][-1] for i in range(len(states))]
    train_ops = tf.group(*(train_W_ops + train_b_ops))

    sess = tf.Session()

    sess.run(tf.initialize_all_variables())

    gen_train, _, gen_test = create_data(batch_size=batch_size)

    acc_train = []
    acc_test = []
    lr = 3e-5

    prev_t = start_t
    for i in range(nepochs):
        gen_inputs, gen_targets = image2spiketrain(*gen_train.next(), max_duration=T)
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
        curr_t = time.perf_counter()
        print(' '.join('{:1.3f}'.format(k) for k in acc_train[-1]) + ' {:.3f}'.format(curr_t-prev_t) + ' seconds')
        if (i%20)==0:
            gen_inputs, gen_targets = image2spiketrain(*gen_test.next(), max_duration=T)
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
            curr_t = time.perf_counter()
            print(' '.join('{:1.3f}'.format(k) for k in acc_test[-1]) + ' {:.3f}'.format(curr_t-prev_t) + ' seconds')
        
        prev_t = curr_t
          
    print("End %s:mnist_benchmark() , running time: {} seconds".format(os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))


if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()
    
#     setup_mnist()
#     mnist_benchmark()
    filenames = setup_nsat()
    run_nsat(filenames)
    
    print("End %s:main() , running time: {} seconds".format(os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))
