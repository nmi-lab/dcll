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
import keras
import h5py

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

NUM_CLASSES = 11
CHUNK_SIZE = 500
TEST_PERCENTAGE = .3
input_height = 240
input_width = 304
input_shape = [input_width * input_height]


def most_common(lst):
    return max(set(lst), key=lst.count)


class sequence_generator(object):
    def __init__(self, targets, data, chunk_size=CHUNK_SIZE, batch_size=32, shuffle=True):
        self.num_classes = 11
        self.i = 0
        data = data[targets != 0]
        targets = targets[targets != 0] - 1
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.n = (len(targets) // batch_size) * batch_size
        self.n_total = (self.n // self.batch_size) // self.chunk_size
        self.x = self.data = data[:self.n]
        self.targets = targets[:self.n]
        self.y = keras.utils.to_categorical(self.targets, self.num_classes)
        self.base = np.array([list(range(self.chunk_size))] * self.batch_size)

    def __iter__(self):
        return self

    def reset(self):
        self.i = 0

    def next(self):
        return self.__next__()


class sequence_generator_sorted(sequence_generator):
    def __init__(self, targets, data, chunk_size=CHUNK_SIZE, batch_size=32, shuffle=True):
        self.num_classes = NUM_CLASSES
        self.targets = targets
        # data_sorted is a list
        # such that data_sorted[0] contains all data sequences for object 0
        # such that data_sorted[1] contains all data sequences for object 1 etc.
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
                idx = np.random.randint(0, self.n[k], size=bk)
                mbx[count:count + bk] = self.data_sorted[k][0][0]
                mbt[count:count + bk] = [k] * bk
                # mbt[count:count + bk] = self.targets[k][0][0] BOUNDINGBOXES
                count += bk
        return mbx, expand_targets(keras.utils.to_categorical(mbt, self.num_classes), self.chunk_size)


def create_data(valid=False, chunk_size=CHUNK_SIZE, batch_size=100):
    dataset = h5py.File('/home/eneftci_local/massiset/massiset.hdf5', 'r') #/Users/massimilianoiacono/Desktop/ripper
    targets_train = [[] for i in range(NUM_CLASSES)]
    data_train = [[] for i in range(NUM_CLASSES)]
    targets_test = [[] for i in range(NUM_CLASSES)]
    data_test = [[] for i in range(NUM_CLASSES)]

    for el in list(dataset.keys()):
        obj_ind = mapping[''.join([i for i in el if not i.isdigit()])]
        targets_train[obj_ind].append(dataset.get(el).get('bbox_train'))
        targets_test[obj_ind].append(dataset.get(el).get('bbox_test'))
        data_train[obj_ind].append(dataset.get(el).get('data_train'))
        data_test[obj_ind].append(dataset.get(el).get('data_test'))

    gen_train = sequence_generator_sorted(
        targets_train,
        data_train,
        chunk_size=chunk_size,
        batch_size=batch_size,
        shuffle=True)

    gen_test = sequence_generator_sorted(
        targets_test,
        data_test,
        chunk_size=chunk_size,
        shuffle=True,
        batch_size=batch_size)

    return gen_train, gen_test


def expand_targets(targets, T=500):
    return np.tile(targets.copy(), [T, 1, 1])


if __name__ == '__main__':
    gen_train, gen_test = create_data(batch_size=50)
    data_batch, target_batch = gen_train.next()
    pass