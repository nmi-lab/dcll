#!/bin/python
# -----------------------------------------------------------------------------
# File Name : dvs_gesture.py
# Author: Emre Neftci
#
# Creation Date : Sat 02 Dec 2017 08:22:08 PM PST


# Last Modified : Mon 23 Apr 2018 09:15:15 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------
import numpy as np
import h5py
from dvs_timeslices import *
from collections import Counter

# actionlabel
mapping = {'backpack': 0,
           'ball': 1,
           'banana': 2,
           'book': 3,
           'bottle': 4,
           'cell_phone': 5,
           'cup': 6,
           'keyboard': 7,
           'mouse': 8,
           'remote': 9,
           'scissors': 10,
           }

mapping2 = {0: 'backpack',
           1: 'ball',
           2: 'banana',
           3: 'book',
           4: 'bottle',
           5: 'cell_phone',
           6: 'cup',
           7: 'keyboard',
           8: 'mouse',
           9: 'remote',
           10: 'scissors',
           }

NUM_CLASSES = 11
CHUNK_SIZE = 250
DELTAT=1000
TEST_PERCENTAGE = .3
DOWNSAMPLE = 4
input_height = 240//DOWNSAMPLE
input_width = 304//DOWNSAMPLE
input_shape = [input_width, input_height]


class abstractSequenceGenerator(object):
    def __iter__(self):
        return self

    def reset(self):
        self.i = 0

    def next(self):
        return self.__next__()

class sequence_generator(abstractSequenceGenerator):
    def __init__(self, targets, data, chunk_size=CHUNK_SIZE, batch_size=32, shuffle=True):
        self.num_classes = NUM_CLASSES
        self.targets = targets
        self.data_sorted = data
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        # number of samples per object class
        self.n = [None for _ in range(self.num_classes)]
        for k in range(self.num_classes):
            self.n[k] = (len(self.data_sorted[k][0]))

        self.prior = np.array(self.n, dtype='float') / np.sum(self.n)
        self.reset()

    def reset(self):
        self.position = [0 for k in range(self.num_classes)]

    def create_sample(self, data, i):
        evs_tmp = data[i:]
        evs = chunk_evs(evs_tmp, deltat = DELTAT, chunk_size = self.chunk_size, size = input_shape, ds = DOWNSAMPLE)
        bx = evs
        return bx

    def __next__(self):
        # data samples
        mbx = np.zeros((self.batch_size, self.chunk_size, input_width, input_height))
        # target samples
        mbt = np.zeros(self.batch_size)
        # mbt = np.zeros((self.batch_size,4)) BOUNDING BOXES

        if self.shuffle:
            # data samples to take
            sequence_size = np.random.multinomial(self.batch_size, self.prior, 1)[0]
        else:
            sequence_size = self.position

        count = 0
        for k, bk in enumerate(sequence_size):
            if bk > 0:
                ids = np.random.randint(0, self.n[k]-self.chunk_size, size=bk)
                mbx[count:count+bk,...] = [self.create_sample(self.data_sorted[k][0], i) for i in ids]
                mbt[count:count+bk] = k 
                count += bk
                    
                # mbt[count:count + bk] = self.targets[k][0][0] BOUNDINGBOXES
        return mbx, expand_targets(one_hot(mbt, self.num_classes), self.chunk_size)


def create_data(valid=False, chunk_size=CHUNK_SIZE, batch_size=100):
    dataset = h5py.File('/share/data/massiset/massiset_sparse.hdf5', 'r') #/Users/massimilianoiacono/Desktop/ripper
    targets_train = [[] for i in range(NUM_CLASSES)]
    data_train = [[] for i in range(NUM_CLASSES)]
    targets_test = [[] for i in range(NUM_CLASSES)]
    data_test = [[] for i in range(NUM_CLASSES)]

    for el in list(dataset.keys()):
        obj_ind = mapping[''.join([i for i in el if not i.isdigit()])]
        targets_train[obj_ind].append(dataset.get(el).get('bbox_train').value)
        targets_test[obj_ind].append(dataset.get(el).get('bbox_test').value)
        data_train[obj_ind].append(dataset.get(el).get('data_train').value)
        data_test[obj_ind].append(dataset.get(el).get('data_test').value)

    gen_train = sequence_generator(
        targets_train,
        data_train,
        chunk_size=chunk_size,
        batch_size=batch_size,
        shuffle=True)

    gen_test = sequence_generator(
        targets_test,
        data_test,
        chunk_size=chunk_size,
        shuffle=True,
        batch_size=batch_size)

    return gen_train, gen_test




if __name__ == '__main__':
    gen_train, gen_test = create_data(batch_size=50)
    data_batch, target_batch = gen_train.next()
    pass
