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
from collections import Counter
            #action,label
mapping = { 0 :'class0',
            1 :'class1',
            2 :'class2',
            3 :'class3',
            4 :'class4',
            5 :'class5',
            6 :'class6',
            7 :'class7',
            8 :'class8',
            9 :'class9',
            10:'class10'}

NUM_CLASSES = 11
CHUNK_SIZE = 500
input_shape = [152*120]

def most_common(lst):
    return max(set(lst), key=lst.count)


class sequence_generator(object):
    def __init__(self, targets, data, chunk_size=CHUNK_SIZE, batch_size=32, shuffle = True):
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
        self.base = np.array([list(range(self.chunk_size))]*self.batch_size)


    def __iter__(self):
        return self

    def reset(self):
        self.i = 0

    def next(self):
        return self.__next__()

class sequence_generator_sorted(sequence_generator):
    def __init__(self, bbox, data, chunk_size=CHUNK_SIZE, batch_size=32, shuffle = True):
        self.num_classes = NUM_CLASSES
        data = data[targets!=0]
        self.bbox = bbox
        self.data_sorted = data
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.n = [None for _ in range(self.num_classes)]
        for k in range(self.num_classes):
            self.n[k] = (len(data_sorted[k])

        self.prior = np.array(self.n, dtype='float')/np.sum(self.n)
        self.reset()

    def reset(self):
        self.position = [0 for k in range(self.num_classes)]
        self.togo = np.concatenate([k*np.ones(self.n[k], dtype='int') for k in range(self.num_classes)])

    def __next__(self):
        mbx = np.zeros([self.batch_size, self.chunk_size]+input_shape)
        mbt = np.zeros([self.batch_size])
        if self.shuffle:
            #data samples to take
            sequence_size = np.random.multinomial(self.batch_size,self.prior,1)[0]
        else:
            raise NotImplementedError()
        count = 0
        if self.shuffle:
            for k,bk in enumerate(sequence_size):
                if bk>0:
                    idx = np.random.randint(0,self.n[k])
                    d = self.data[k][idx].reshape(-1,self.chunk_size,input_shape[0])
                    mbx[count:count+bk] = d
                    mbt[count:count+bk] = bbox[k][idx]
        return mbx, expand_targets(keras.utils.to_categorical(mbt, self.num_classes), self.chunk_size)

def create_data(valid = False, chunk_size=CHUNK_SIZE, batch_size=100):
    targets_train = np.load('dataset/train_pol_streaming1000_labels.npy')
    data_train = np.load('dataset/train_pol_streaming1000.npy')
    gen_train = sequence_generator_sorted(
            targets_train,
            data_train,
            chunk_size = chunk_size,
            batch_size = batch_size,
            shuffle = True)

    targets_test = np.load('dataset/test_pol_streaming1000_labels.npy')
    data_test = np.load('dataset/test_pol_streaming1000.npy')
    gen_test = sequence_generator_sorted(
            targets_test,
            data_test,
            chunk_size = chunk_size,
            shuffle=True,
            batch_size = batch_size)

    return gen_train, gen_test  

def expand_targets(targets,T=500):
    return np.tile(targets.copy(),[T,1,1])
    
if __name__ == '__main__':
    gen_train, gen_test = create_data(batch_size=50)





