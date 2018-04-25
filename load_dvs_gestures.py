#!/bin/python
#-----------------------------------------------------------------------------
# File Name : dvs_gesture.py
# Author: Emre Neftci
#
# Creation Date : Sat 02 Dec 2017 08:22:08 PM PST
# Last Modified : Mon 23 Apr 2018 09:15:15 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
import keras
            #action,label
mapping = { 0 :'Hand Clapping'  ,
            1 :'Right Hand Wave',
            2 :'Left Hand Wave' ,
            3 :'Right Arm CW'   ,
            4 :'Right Arm CCW'  ,
            5 :'Left Arm CW'    ,
            6 :'Left Arm CCW'   ,
            7 :'Arm Roll'       ,
            8 :'Air Drums'      ,
            9 :'Air Guitar'     ,
            10:'Other'}


input_shape = [1024]

def most_common(lst):
    return max(set(lst), key=lst.count)


class sequence_generator(object):
    def __init__(self, targets, data, chunk_size=500, batch_size=32, shuffle = True):
        self.num_classes = 11
        self.i = 0
        data = data[targets!=0]
        targets = targets[targets!=0]-1
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.n = (len(targets)//batch_size)*batch_size
        self.n_total = (self.n//self.batch_size)//self.chunk_size
        self.x = self.data = data[:self.n]
        self.targets = targets[:self.n]
        self.y = keras.utils.to_categorical(self.targets, self.num_classes)
        self.base = np.array([range(self.chunk_size)]*self.batch_size)


    def __iter__(self):
        return self

    def reset(self):
        self.i = 0

    def next(self):
        #Randomize order of learning
        if self.shuffle:
            idx = (self.base.T+np.random.randint(0,self.n-self.chunk_size,size=self.batch_size)).T
        else:
            idx = (self.base.T+np.linspace(0,self.n-(self.n)/self.batch_size-self.chunk_size,self.batch_size, dtype='uint32')+self.i).T
            self.i+=self.chunk_size
            self.i = self.i%((self.n)/self.batch_size)
        d = self.data[idx].swapaxes(1,3)
        t = self.targets[idx]
        return d, keras.utils.to_categorical(t, self.num_classes)

#Experiment Parameters
batch_size = 100
nepochs = 200

def create_data(batch_size=100, small=False):
    targets_test = np.load('/share/data/DvsGesture/test_pol_streaming1000_labels.npy')
    data_test = np.load('/share/data/DvsGesture/test_pol_streaming1000.npy')
    gen_test = sequence_generator(targets_test, data_test, shuffle=True, batch_size = batch_size)

    if not small:
        targets_train = np.load('/share/data/DvsGesture/train_pol_streaming1000_labels.npy')
        data_train = np.load('/share/data/DvsGesture/train_pol_streaming1000.npy')
        gen_train = sequence_generator(targets_train, data_train, shuffle=True, batch_size = batch_size )
    else:
        gen_train=gen_test

    return gen_train, gen_test  

def expand_targets(targets,T=500):
    return np.tile(targets.copy(),[T,1,1])
    






