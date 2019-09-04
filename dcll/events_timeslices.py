#!/bin/python
#-----------------------------------------------------------------------------
# File Name : event_timeslices.py
# Author: Emre Neftci
#
# Creation Date : 
# Last Modified : Thu 16 May 2019 02:13:09 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from __future__ import print_function
import bisect
import numpy as np
from scipy.sparse import coo_matrix as sparse_matrix

def expand_targets(targets, T=500, burnin=0):
    y = np.tile(targets.copy(), [T, 1, 1])
    y[:burnin] = 0
    return y

def one_hot(mbt, num_classes):
    out = np.zeros([mbt.shape[0], num_classes])
    out[np.arange(mbt.shape[0], dtype='int'),mbt.astype('int')] = 1
    return out

def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)

def cast_evs(evs):
    ts = (evs[:,0]*1e6).astype('uint64')
    ad = (evs[:,1:]).astype('uint64')
    return ts, ad

def find_idx_time(evs, idx):
    return idx_end

def get_binary_frame(evs, size = (346,260), ds=1):
    tr = sparse_matrix((2*evs[:,3]-1,(evs[:,1]//ds,evs[:,2]//ds)), dtype=np.int8, shape=size)
    return tr.toarray()

def get_binary_frame_np(arr, evs, size = (346,260), ds=1):
    arr[evs[:,1]//ds,evs[:,2]//ds] = 2*evs[:,3]-1

def get_event_timeslice(device, Deltat = 1000*50):
    #flush
    device.get_event()
    t = -1
    evs = []
    while t<Deltat:
        evs_frame_tmp = device.get_event()[0]
        evs.append(evs_frame_tmp)
        if t == -1:
            t0 = evs_frame_tmp[0,0]
        try:
            t = (evs_frame_tmp[-1,0] - t0)
        except TypeError:
            continue


    evs = np.row_stack(evs)
    idx_end = np.searchsorted(evs[:,0], t0+Deltat)
    evs_frame = evs[:idx_end]
    evs_frame[:,0]= -evs_frame[-1,0] + evs_frame[:,0]
    print(evs_frame[0,0],evs_frame[-1,0])
    return evs_frame

def get_time_surface(device, invtau = 1e-6, size= (346,260,2)):
    evs = get_event_timeslice(device)

    tr = np.zeros(size, 'int64')-np.inf

    for ev in evs:
        tr[ev[1],ev[2],ev[3]]=ev[0]

    a = np.exp(tr[:,:,0]*1e-6)-np.exp(tr[:,:,1]*1e-6)
    return a

def chunk_evs(evs, deltat=1000, chunk_size=500, size = [304,240], ds = 1):
    t_start = evs[0,0]
    ts = range(t_start, t_start + chunk_size*deltat, deltat)
    chunks = np.zeros([len(ts)]+size, dtype='int8')
    idx_start=0
    idx_end=0
    for i,t in enumerate(ts):
        idx_end += find_first(evs[idx_end:,0], t)
        if idx_end>idx_start:
            get_binary_frame_np(chunks[i, ...], evs[idx_start:idx_end], size=size, ds = ds)
        idx_start = idx_end
    return chunks

def chunk_evs_pol(times, addrs, deltat=1000, chunk_size=500, size = [2,304,240], ds = 1):
    t_start = times[0]
    ts = range(t_start, t_start + chunk_size*deltat, deltat)
    chunks = np.zeros([len(ts)]+size, dtype='int8')
    idx_start=0
    idx_end=0
    for i,t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t)
        if idx_end>idx_start:
            ee = addrs[idx_start:idx_end]
            pol,x,y = ee[:,2],ee[:,0]//ds,ee[:,1]//ds
            np.add.at(chunks,(i,pol,x,y),1) 
        idx_start = idx_end
    return chunks



if __name__ == "__main__":
    import h5py
    dataset = h5py.File('/home/eneftci_local/Projects/share/data/massiset/massiset_sparse.hdf5', 'r')
    evs = dataset.get('backpack')['data_train'].value
    cevs = chunk_evs(evs,chunk_size=500,deltat=1000, ds = 4, size = [304//4, 240//4])
