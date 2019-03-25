import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms, utils
import torch
from torch.utils.data.dataloader import _DataLoaderIter, DataLoader
import tqdm

def __gen_ST(N, T, rate, mode = 'regular'):    
    if mode == 'regular':
        spikes = np.zeros([T, N])
        spikes[::(1000//rate)] = 1
        return spikes
    elif mode == 'poisson':
        spikes = np.ones([T, N])        
        spikes[np.random.binomial(1,float(1000. - rate)/1000, size=(T,N)).astype('bool')] = 0
        return spikes
    else:
        raise Exception('mode must be regular or Poisson')
        
def spiketrains(N, T, rates, mode = 'poisson'):
    '''
    *N*: number of neurons
    *T*: number of time steps
    *rates*: vector or firing rates, one per neuron
    *mode*: 'regular' or 'poisson'
    '''
    if not hasattr(rates, '__iter__'):
        return __gen_ST(N, T, rates, mode)
    rates = np.array(rates)
    M = rates.shape[0]
    spikes = np.zeros([T, N])
    for i in range(M):
        if int(rates[i])>0:
            spikes[:,i] = __gen_ST(1, T, int(rates[i]), mode = mode).flatten()
    return spikes

def spikes_to_evlist(spikes):
    t = np.tile(np.arange(spikes.shape[0]), [spikes.shape[1],1])
    n = np.tile(np.arange(spikes.shape[1]), [spikes.shape[0],1]).T  
    return t[spikes.astype('bool').T], n[spikes.astype('bool').T]

def plotLIF(U, S, Vplot = 'all', staggering= 1, ax1=None, ax2=None, **kwargs):
    '''
    This function plots the output of the function LIF.
    
    Inputs:
    *S*: an TxNnp.array, where T are time steps and N are the number of neurons
    *S*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    *Vplot*: A list indicating which neurons' membrane potentials should be 
    plotted. If scalar, the list range(Vplot) are plotted. Default: 'all'
    *staggering*: the amount by which each V trace should be shifted. None
    
    Outputs the figure returned by figure().    
    '''
    V = U
    spikes = S
    #Plot
    t, n = spikes_to_evlist(spikes)
    #f = plt.figure()
    if V is not None and ax1 is None:
        ax1 = plt.subplot(211)
    elif V is None:
        ax1 = plt.axes()
        ax2 = None
    ax1.plot(t, n, 'k|', **kwargs)
    ax1.set_ylim([-1, spikes.shape[1] + 1])
    ax1.set_xlim([0, spikes.shape[0]])

    if V is not None:
        if Vplot == 'all':
            Vplot = range(V.shape[1])
        elif not hasattr(Vplot, '__iter__'):
            Vplot = range(np.minimum(Vplot, V.shape[1]))    
        
        if ax2 is None:
            ax2 = plt.subplot(212)
    
        if V.shape[1]>1:
            for i, idx in enumerate(Vplot):
                ax2.plot(V[:,idx]+i*staggering,'-',  **kwargs)
        else:
            ax2.plot(V[:,0], '-', **kwargs)
            
        if staggering!=0:
            plt.yticks([])
        plt.xlabel('time [ms]')
        plt.ylabel('u [au]')

    ax1.set_ylabel('Neuron ')

    plt.xlim([0, spikes.shape[0]])
    plt.ion()
    plt.show()
    return ax1,ax2


import numpy as np

input_shape = [28,28,1]


def to_one_hot(t, width):
    t_onehot = torch.zeros(*t.shape+(width,))
    return t_onehot.scatter_(1, t.unsqueeze(-1), 1)

class DataLoaderIterPreProcessed(_DataLoaderIter):
    def __next__(self):
       indices = next(self.sample_iter)  # may raise StopIteration
       batch = self.collate_fn([self.dataset.preprocessed_get(i) for i in indices])
       if self.pin_memory:
           batch = pin_memory_batch(batch)
       return batch

class DataLoaderPreProcessed(DataLoader):
    def __iter__(self):
        return DataLoaderIterPreProcessed(self)

def preprocess_dataset(dataset):
    iterit = iter(dataset)
    x, y = dataset[0]
    td = torch.empty(torch.Size([len(dataset)])+x.shape, dtype = x.dtype)
    tl = torch.empty(torch.Size([len(dataset)])+y.shape, dtype = y.dtype)
    for idx in tqdm.tqdm(range(len(dataset)), desc = "Pre-processing dataset"):
        td[idx], tl[idx] = dataset[idx]
        
    if dataset.train:
        dataset.train_data, dataset.train_labels = td, tl
    else:
        dataset.test_data, dataset.test_labels = td, tl

    def get(idx):
        if dataset.train:
            return dataset.train_data[idx], dataset.train_labels[idx] 
        else:
            return dataset.test_data[idx], dataset.test_labels[idx] 
        
    dataset.preprocessed_get = get
    dataset.transform = None
    return dataset

def pixel_permutation(d_size, r_pix=1.0, seed=0):
   import copy
   n_pix = int(r_pix * d_size)
   np.random.seed(seed*131313)
   pix_sel = np.random.choice(d_size, n_pix, replace=False).astype(np.int32)
   pix_prm = np.copy(pix_sel)
   np.random.shuffle(pix_prm)
   perm_inds = np.arange(d_size)
   perm_inds[pix_sel] = perm_inds[pix_prm]
   return perm_inds

def permute_dataset(dataset, r_pix, seed):
    if hasattr(dataset, 'train_data'):
        datap = dataset.train_data
    elif hasattr(dataset, 'test_data'):
        datap = dataset.test_data
    else:
        raise Exception('no data found')
    perm = pixel_permutation(np.prod(datap.shape[1:]), r_pix, seed=seed)
    orig_shape = datap.shape[1:]
    datap = datap.view(-1, np.prod(datap.shape[1:]))[:,perm].view(-1,*orig_shape)
    if hasattr(dataset, 'train_data'):
        dataset.train_data = datap
    elif hasattr(dataset, 'test_data'):
        dataset.test_data = datap

def partition_dataset(dataset, Nparts=600, part=0):
    if hasattr(dataset, 'train_data'):
        datap = dataset.train_data
        labelsp = dataset.train_labels
    elif hasattr(dataset, 'test_data'):
        datap = dataset.test_data
        labelsp = dataset.test_labels
    else:
        raise Exception('no data found')

    N = len(datap)
    step = (N//Nparts)
    datap = datap[step*part:step*(part+1)]
    labelsp = labelsp[step*part:step*(part+1)]

    if hasattr(dataset, 'train_data'):
        dataset.train_data = datap
        dataset.train_labels = labelsp
    elif hasattr(dataset, 'test_data'):
        dataset.test_data = datap
        dataset.test_labels = labelsp

def get_mnist_loader(batch_size, train, perm=0., Nparts=1, part=0, seed=0, taskid=0, pre_processed=True, **loader_kwargs):
    """Builds and returns Dataloader for MNIST dataset."""
    
    transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.0,), (1.0,)),
                    transforms.Lambda(lambda x: x.view([28,28]))])

    dataset = datasets.MNIST(root='./data', download=True, transform=transform, train = train)
    if perm>0:
        permute_dataset(dataset, perm, seed=seed)
    if Nparts>1:
        partition_dataset(dataset, Nparts,part)

    if pre_processed:
        dataset = preprocess_dataset(dataset)
        DL = DataLoaderPreProcessed
    else:
        DL = DataLoader
    
    loader = DL(dataset=dataset,
                batch_size=batch_size,
                shuffle=train,
                **loader_kwargs)
    loader.taskid = taskid
    loader.name = 'MNIST_{}'.format(taskid,part)
    loader.short_name = 'MNIST'
    return loader

def image2spiketrain(x,y,gain=50,min_duration=None, max_duration=500):
    y = to_one_hot(y, 10)
    if min_duration is None:
        min_duration = max_duration-1
    batch_size = x.shape[0]
    T = np.random.randint(min_duration,max_duration,batch_size)
    Nin = np.prod(input_shape)
    allinputs = np.zeros([batch_size,max_duration, Nin])
    for i in range(batch_size):
        st = spiketrains(T = T[i], N = Nin, rates=gain*x[i].reshape(-1)).astype(np.float32)
        allinputs[i] =  np.pad(st,((0,max_duration-T[i]),(0,0)),'constant')
    allinputs = np.transpose(allinputs, (1,0,2))
    allinputs = allinputs.reshape(allinputs.shape[0],allinputs.shape[1],1, 28,28)

    alltgt = np.zeros([max_duration, batch_size, 10], dtype=np.float32)
    for i in range(batch_size):
        alltgt[:,i,:] = y[i]

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
