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

class sequence_generator_sorted(object):
    def __init__(self, targets, data, chunk_size=500, batch_size=32, shuffle = True):
        self.num_classes = 11
        data = data[targets!=0]
        targets = targets[targets!=0]-1
        data_sorted = sort_by_class(data, targets)
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.n = [None for _ in range(self.num_classes)]
        for k in range(self.num_classes):
            self.n[k] = (len(data_sorted[k])//chunk_size)*chunk_size
        self.n_total = (np.sum(np.array(self.n,'int')//self.chunk_size)//self.batch_size)
        self.x = self.data = [data_sorted[k][:self.n[k]] for k in range(self.num_classes)]
        self.base = np.array([range(self.chunk_size)]*self.batch_size)
        self.prior = np.array(self.n, dtype='float')/np.sum(self.n)
        self.reset()

    def reset(self):
        self.position = [0 for k in range(self.num_classes)]
        self.togo = np.concatenate([k*np.ones(self.n[k], dtype='int') for k in range(self.num_classes)])
        self.pointer = 0

    def next(self):
        mbx = np.zeros([self.batch_size,self.chunk_size]+input_shape)
        mbt = np.zeros([self.batch_size])
        if self.shuffle:
            sequence_size = np.random.multinomial(self.batch_size,self.prior,1)[0]
        else:
            from collections import Counter
            if self.pointer+self.batch_size*self.chunk_size>len(self.togo):
                raise Exception("data exhausted, run reset, then next up to self.ntotal times")
            sequence_size = Counter(self.togo[self.pointer:self.pointer+self.batch_size*self.chunk_size])
            self.pointer+=self.batch_size*self.chunk_size
        count = 0
        if self.shuffle:
            for k,bk in enumerate(sequence_size):
                if bk>0:
                    idx = (self.base.T[:,count:count+bk]+self.chunk_size*np.random.randint(0,self.n[k]//self.chunk_size-bk,size=bk)).T
                    d = self.x[k][idx].reshape(-1,self.chunk_size,input_shape[0])
                    mbx[count:count+bk] = d
                    mbt[count:count+bk] = k
                    count+=bk

        else:
            for k,bk in sequence_size.items():
                bk = bk/self.chunk_size
                idx = np.arange(0,bk*self.chunk_size, dtype='uint32').reshape(bk,self.chunk_size)
                #idx = (self.base.T[:,count:count+bk]+new_idx+self.position[k]).T
                self.position[k] += bk*self.chunk_size
                d = self.x[k][idx].reshape(-1,self.chunk_size,input_shape[0])
                mbx[count:count+bk] = d
                mbt[count:count+bk] = k
                count+=bk
        assert count == self.batch_size
        return mbx, expand_targets(keras.utils.to_categorical(mbt, self.num_classes), self.chunk_size)

def plot_gestures_imshow(gen, nim=11):
    import pylab as plt
    plt.figure(figsize = [nim+2,6])
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(6, nim)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=.0, hspace=.04)
    images, labels = gen.next()
    categories = labels.argmax(axis=1)
    idx = 0
    for j in range(nim):
         #idx = np.where(categories==j)[0][0]
         idx += 1 
         for i in range(6):
             ax = plt.subplot(gs[i, j])
             plt.imshow(images[idx,:,:,i].T)
             plt.xticks([])
             if i==0:  plt.title(mapping[labels[idx].argmax()], fontsize=10)
             plt.yticks([])
             plt.bone()
    return images,labels
            


def sort_by_class(data, targets):
    data_sorted = [None for _ in range(11)]
    for i in range(11):
        idx = targets == i
        data_sorted[i] = data[idx]
    return data_sorted

#Experiment Parameters
batch_size = 100
nepochs = 200

def create_data(valid = False, batch_size=100, sorted=True):
    targets_train = np.load('/share/data/DvsGesture/train_pol_streaming1000_labels.npy')
    data_train = np.load('/share/data/DvsGesture/train_pol_streaming1000.npy')
    if sorted:
        gen_train = sequence_generator_sorted(targets_train, data_train, batch_size = batch_size)
    else:
        gen_train = sequence_generator(targets_train, data_train, batch_size = batch_size)

    targets_test = np.load('/share/data/DvsGesture/test_pol_streaming1000_labels.npy')
    data_test = np.load('/share/data/DvsGesture/test_pol_streaming1000.npy')
    if sorted:
        gen_test = sequence_generator_sorted(targets_test, data_test, shuffle=True, batch_size = batch_size)
    else:
        gen_test = sequence_generator(targets_test, data_test, shuffle=True, batch_size = batch_size)
    return gen_train, gen_test  

def expand_targets(targets,T=500):
    return np.tile(targets.copy(),[T,1,1])
    
if __name__ == '__main__':
    gen_train, gen_test = create_data(batch_size=50)





