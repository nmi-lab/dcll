#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File Name : npamlib.py
# Author: Emre Neftci, Ryan Stokes
#
# Creation Date : 03-03-2016
# Last Modified : Fri 04 Nov 2016 12:44:30 AM PDT
#
# Copyright : (c) UC Regents, Emre Neftci, Ryan Stokes
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib,sys
matplotlib.rcParams['savefig.dpi']=100
matplotlib.rcParams['font.size']=12.0
matplotlib.rcParams['font.weight']='bold'



def stitch_features(filters,rows=8, margin = 1):
  cols = filters.shape[2]/rows
  w = filters.shape[0]
  h = filters.shape[1]
  m = margin
  stitched = [None for i in range(rows)]
  for i in range(rows):
    imgs = []
    for j in range(cols):
      imgpad = np.zeros([w+m,w+m])
      imgpad[:w,:h] = filters[:,:,i*cols + j]
      imgs.append(imgpad)
      stitched[i] = np.column_stack(imgs)
  return np.row_stack(stitched)

def stitch_rgb_features(w):
    wc = np.array([stitch_features(w[:,:,i,:]) for i in range(3)])
    return np.swapaxes(wc,0,2)


def data_load_mnist(digits = None):
    '''
    Download and load MNIST hand-written digits
    Inputs:
    *digits*: list specifying which digits should be returned (default: return all digits 0-9)
    Ouputs: 
    *images* a 1000x784np.array storing 1000 28x28 pixel images of hand-written
    digits
    *labels* labels of the 1000 images
    '''
    import os.path
    url_data = "https://drive.google.com/uc?export=download&id=0B7CeL_WOYFxpTl94RHkxN0pfMEk"
    url_targets = "https://drive.google.com/uc?export=download&id=0B7CeL_WOYFxpZmY4T1hYSHJNTjQ"
    fname_data = 'mnist_data.npy'
    fname_targets = 'mnist_targets.npy'
    if sys.version_info[0] == 2:
        if not os.path.isfile(fname_data):
            import urllib
            urllib.urlretrieve(url_data, fname_data)
        if not os.path.isfile(fname_targets):
            import urllib
            urllib.urlretrieve(url_targets, fname_targets)
    elif sys.version_info[0] == 3:
        if not os.path.isfile(fname_data):
            import urllib.request
            urllib.request.urlretrieve(url_data, fname_data)
        if not os.path.isfile(fname_targets):
            import urllib.request
            urllib.request.urlretrieve(url_targets, fname_targets)
    data = np.load(fname_data)
    labels = np.load(fname_targets)
    if digits is None:
        return data, labels 
    else:
        idx = np.zeros_like(labels, dtype = 'bool')
        for d in digits:
            idx+= labels == d
        return data[idx,:], labels[idx]

def one_hot(labels, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(labels).reshape(-1)
    return np.eye(nb_classes)[targets]

def relu(r):
    return np.maximum(r,0)

def threshold(r):
    rr = np.empty_like(r)
    rr[r<=0]=0
    rr[r>0]=1
    return rr

def sigmoid(r):
    return 1./(1+np.exp(-r))

def softmax(x):
    return np.exp(x)/np.expand_dims(np.sum(np.exp(x),axis=1),axis=1)

def softplus(r, alpha = 1):
    return np.log(1+np.exp(alpha * r))

def softplus_arp(r, alpha = 1, tarp = 1):
    sp = softplus(r, alpha = alpha)
    return sp/(sp+tarp)

def sigmoid_prime(r):
    """Derivative of the sigmoid function."""
    return sigmoid(r)*(1-sigmoid(r))

def LIF(N=32, T=1000, alpha=.95, alphaS=0.975, theta=1, tarp=0, b=0.0, Win = None, W = None, spikes_in = None):
    '''
    Simulates a network of leaky integrate and fire neurons.
    Inputs:
    *N* Number of neurons (default: 32)
    *T* Number of time steps to simulate (default: 1000)
    *alpha* Integration time constant of the membrane potential (default .95)
    *alphas* integration time constant of the synaptic current  (default .975)
    *theta* Firing threshold (default 1)
    *tarp* Absolute Refractory Period (defualt 0)
    *W* Recurrent weight matrix (default None)
    *Win* Input weight matrix (default np.eye(N))
    *spikes_in* input spikes as a (T,N)np.array (default None)

    Outputs:
    *V* membrane potentials as a [T,N] numpynp.array
    *spikes* spikes as a [T,N] numpynp.array
    '''
    #Variables
    V = np.zeros([T, N])
    dtarp = np.zeros([N])

    #Random initial state

    Isyn = np.zeros([N])
    spikes = np.zeros([T, N])

    if Win is None and spikes_in is not None:
        Win = np.eye(N)

    if spikes_in is not None:
        spikes_in = spikes_in.reshape([T,-1])

    if not hasattr(b, '__len__'):
        b = np.ones([N])*b

    #Main loop
    tV = np.random.rand(N)
    for t in range(T): 
        tIsyn = Isyn.copy()

        id_non_refr = dtarp == 0
        tV[~id_non_refr] = 0 
        tV[id_non_refr] = alpha*tV[id_non_refr] + b[id_non_refr] + Isyn[id_non_refr]

        tIsyn = alphaS*Isyn 

        if Win is not None and spikes_in is not None:
            tIsyn += np.dot(spikes_in[t], Win)
        id_spiked = tV>=theta
        tV[id_spiked] = 0     
        if np.any(id_spiked):
            dtarp[id_spiked] = tarp
            spikes[t, id_spiked] = 1 
        dtarp[~id_non_refr] -= 1

        if W is not None:
            tIsyn += np.dot(spikes[t], W)

        V[t] = tV
        Isyn = tIsyn


    return V, spikes
    
def FRN(N=32, b=0.0, Win = None, rates_in = None, activation_function = relu):
    '''
    Simulates a network of firing rate neurons.
    Inputs:
    *N* Number of neurons (default: 32)
    *activation_function* activation function of the neuron
    *Win* Input weight matrix (default np.eye(N))
    *rates_in* input rates as a (N)np.array (default None)

    Outputs:
    *rates* rates as a [T,N] numpynp.array
    '''
    #Variables

    #Random initial state

    tIsyn = np.zeros([N])
    if Win is None and rates_in is not None:
        Win = np.eye(N)

    if not hasattr(b, '__len__'):
        b = np.ones([N])*b

    #Main loop
    tV = np.zeros([N])

    if Win is not None and rates_in is not None:
        tIsyn += np.dot(rates_in, Win)
        
    tV = activation_function(b + tIsyn)

    return tV
    
def __spikes_to_evlist(spikes):
    t = np.tile(np.arange(spikes.shape[0]), [spikes.shape[1],1])
    n = np.tile(np.arange(spikes.shape[1]), [spikes.shape[0],1]).T  
    return t[spikes.astype('bool').T], n[spikes.astype('bool').T]
    
def plot_spikes(spikes):
    '''
    This function plots spikes.
    
    Inputs:
    *spikes*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF or generate_spike_trains
    '''
    return plotLIF(spikes)

def plotLIF(V, spikes, Vplot = 'all', staggering= 1, ax1=None, ax2=None, **kwargs):
    '''
    This function plots the output of the function LIF.
    
    Inputs:
    *V*: an TxNnp.array, where T are time steps and N are the number of neurons
    *spikes*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    *Vplot*: A list indicating which neurons' membrane potentials should be 
    plotted. If scalar, the list range(Vplot) are plotted. Default: 'all'
    *staggering*: the amount by which each V trace should be shifted. None
    
    Outputs the figure returned by figure().    
    '''
    #Plot
    t, n = __spikes_to_evlist(spikes)
    #f = plt.figure()
    if V is not None and ax1 is None:
        ax1 = plt.subplot(211)
    elif V is None:
        ax1 = plt.axes()
        ax2 = None
    ax1.plot(t, n, 'k.', **kwargs)
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
        plt.xlabel('t [au]')
        plt.ylabel('V [au]')
    else:
        plt.xlabel('time [ms]')
        plt.ylabel('Neuron ')

    plt.xlim([0, spikes.shape[0]])
    plt.ion()
    plt.show()
    return ax1,ax2
    
def plotFRN(rates):
    '''
    This function plots the output of the function FRN.
    
    Inputs:
    *rates*: an TxNnp.array indicating firing rates. This is the output
    returned by FRN

    Outputs the figure returned by figure().    
    '''
    #Plot
    
    f = plt.figure()
    
    plt.imshow(rates.T, aspect='auto')
    plt.ylim([-1, rates.shape[1] + 1])
    plt.xlim([0, rates.shape[0]])
    plt.colorbar()
    return f
    
def plot_spike_count(spikes, average = False):
    rates = spikes.sum(axis=0)
    if average:
        rates /= spikes.shape[0]
    f = plt.figure()
    plt.plot(rates, color = 'k', linewidth=3)
    plt.xlabel('Neuron')
    plt.ylabel('Spike Count')
    plt.show()
    return f
    
def plotAF(inp, spikes):
    '''
    This function plots the activation function of the neurons in LIF.
    
    Inputs:
    *inp*: an Nnp.array indicating the input rate of the neurons. This could be the
    parameter b of function LIF, or the rate of the spike trains in spikes_in
    *spikes*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    Outputs the figure returned by figure().    
    '''
    fig = plt.figure()
    plt.plot(inp,spikes.mean(axis=0),'k.-', linewidth=3)
    plt.ylabel('Avg. Firing Rate [1/tick]')
    plt.xlabel('Input')
    plt.tight_layout()
    return fig
    
def __inst_firing_rate(spikes, window = 100):
    inst_rates = np.empty_like(spikes)
    if not hasattr(window, '__iter__'):
        window = np.ones(window)/window
    for i in range(spikes.shape[1]):
        inst_rates[:,i] = np.convolve(window, spikes[:,i], mode='same')
    return inst_rates
    
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
    if not hasattr(rates, '__iter__'):
        return __gen_ST(N, T, rates, mode)
    rates = np.array(rates)
    M = rates.shape[0]
    spikes = np.zeros([T, N])
    for i in range(M):
        if int(rates[i])>0:
            spikes[:,i] = __gen_ST(1, T, int(rates[i]), mode = mode).flatten()
    return spikes
        
    
def __stim_rotate(image, angle=45):
    from scipy import ndimage
    return ndimage.rotate(image, angle, reshape=False)
    
def stim_vertical_bar(npixels=28, width=4):
    image = np.zeros([npixels, npixels])
    image[:,(npixels/2-width/2):(npixels/2+width/2)]=1
    return image
    
def stim_orientations(image, orientations = 9, flattened = True):
    if not hasattr(orientations, '__iter__'):
        orientations = np.linspace(0, 360, orientations)
    stimuli = [__stim_rotate(image, angle) for angle in orientations]
    if flattened:
        stimuli = [s.flatten() for s in stimuli]
    return np.array(stimuli)

def __tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=False):
    """
    From deeplearning.net
    Transform annp.array with one flattened image per row, into annp.array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-Dnp.array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns:np.array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-dnp.array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = __tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = __scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # outputnp.array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array
        
        
def __scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def stim_show(images):
    '''
    Plots every image in images (using imshow)
    '''
    til = __tile_raster_images(images, 
                    np.array([images.shape[1]**.5, images.shape[1]**.5], 'int'),
                    np.array([images.shape[0]**.5, images.shape[0]**.5], 'int'),
                    tile_spacing = (5,5))

    f = plt.imshow(til)
    plt.bone()
    plt.xticks([]), plt.yticks([])
    plt.show()
    return f
    
    
def ann_createDataSet(N):
    '''
    Random linearly separated data (2 dimensions)
    Inputs:
    *N*: number of samples
    '''
    xA,yA,xB,yB = [np.random.uniform(-1, 1) for i in range(4)]
    yB=.0
    xB=.0
    V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
    X = []
    S = []
    for i in range(N):
        x1,x2 = [np.random.uniform(-1, 1) for i in range(2)]
        x = np.array([x1,x2])
        s = int(np.sign(V[1:].T.dot(x)+V[0]))
        X.append(x)
        S.append(s)
    return np.array(X), np.array(S)  
        
        
def ann_plotSet(X, S, vec = None, f = None):
    '''
    Plot 2D data
    Inputs:
    *X*: data
    *S*: labels
    *vec*: parameter vector
    *f* : figure
    '''
    if f is None:
        plt.figure(figsize=(5,5))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    #a, b = -V[1]/V[2], -V[0]/V[2]
    l = np.linspace(-1,1)
    #plt.plot(l, a*l+b, 'k-')
    cols = {1: 'r', -1: 'b'}
    for x,s in list(zip(X,S)):
        plt.plot(x[0], x[1], cols[s]+'o')
    if vec is not None:
        aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
        plt.plot(l, aa*l+bb, 'g-', lw=2)

 
def ann_classification_error(X, S, vec, pts=None):
    '''
    Compute error defined as fraction of misclassified points
    Inputs:
    *X*: data
    *S*: labels
    *vec*: parameter vector
    *pts*: (default None)
    '''
    if not pts:
        pts = list(zip(X,S))
    M = len(pts)
    n_mispts = 0
    for x,s in pts:
        if int(np.sign(vec[1:].T.dot(x)+vec[0])) != s:
            n_mispts += 1
    error = n_mispts / float(M)
    return error
 
 
def ann_choose_miscl_point(X, S, vec):
    '''
    Choose a random point among the misclassified
    Inputs:
    *X*: data
    *S*: labels
    *vec*: parameter vector
    '''
    mispts = []
    for x,s in zip(X,S):
        if int(np.sign(vec[1:].T.dot(x)+vec[0])) != s:
            mispts.append((x, s))
    return mispts[np.random.randint(0,len(mispts))]
 
 
def ann_demo_pla_2D(N):
    '''
    Demonstration of the Perceptron Learning Algorithm, step by step
    *Inputs*:
    *N*: number of data samples
    '''
    X, S = ann_createDataSet(N)
    # Initialize the weigths to zeros
    w = np.zeros(3)
    it = 0
    f = plt.figure()
    plt.title('%s Iterations' %str(it))
    plt.ion()
    plt.show()
    ann_plotSet(X,S, vec=None, f=f)
    # Iterate until all points are correctly classified
    while True:
        it += 1
        # Pick random misclassified point
        x, s = ann_choose_miscl_point(X,S,w)
        # Update weights
        w[1:] += s*x
        w[0] = s
        if sys.version_info[0] == 2:
            raw_input(':')
        else:
            input(':')
        plt.gca().clear()
        ann_plotSet(X,S, vec=w, f=f)
        plt.title('%s Iterations' %str(it))
        plt.draw()
        if ann_classification_error(X, S, w) == 0:
            break
              
def ann_train_perceptron(data, labels, n, eta, w = None):
    '''
    Train a perceptron on arbitrary data
    Inputs:
    *data*: 2D np.array, one data sample per row
    *labels*: 1D np.array of labels. Labels must be boolean (True/False) or binary
    *n*: number of training steps
    *eta*: learning rate
    *w*: initial weight vector. First term is bias

    Outputs:
    *w*: weight vector after n iterations
    *res*: number of correctly classified points
    '''
    assert labels.dtype == 'bool', "Labels must be boolean (True/False)"
    if w is None:
        w = np.random.rand(1+len(data[0]))
    errors = []
    
    training_data = list(zip(data,labels))

    for i in range(n):
        ii = np.random.choice(len(training_data))
        x, expected = data[ii], labels[ii] 
        result = np.dot(w[1:], x) + w[0]
        error = expected - threshold(result)
        errors.append(error)
        w[1:] += eta * error * x
        w[0] += eta * error

    print('Percent correct:')
    res = np.sum(labels == ann_perceptron(data, w)).astype('float')/len(data)
    print(res)
    

    return w, res

def ann_perceptron(data, w):
    return threshold(np.dot(w[1:], data.T) + w[0]).astype('int')

def ann_train_mlp(data, labels, n, eta, w = None, size = None):
    '''
    Train a single layer MLP on provided data
    Inputs:
    *data*: 2D numpynp.array, one data sample per row
    *labels*: 1D numpynp.array of labels. Labels must be integers
    *n*: number of training steps
    *eta*: learning rate
    *w*: initial weight vector
    *size*: The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. Note that the size of the first layer
        must correspond to the number of features in the data sample and 
        the size of the last layer must correspond to the number of different
        labels.

    Outputs:
    *w*: weight vector after n iterations
    *res*: value of the cost function evaluated on the data
    '''
    if w is not None:
        biases = w[0]
        weights = w[1]
    else:
        biases = None
        weights = None
    if size is None:
        size = [data.shape[1], len(np.unique(labels))]
    else:
        assert size[0] == data.shape[1], "The size of the first layer must correspond to the number of features in the data sample"
        assert size[-1] == len(np.unique(labels)), "The size of the last layer must correspond to the number of different labels"
    mlp = MLPNetwork(size, weights = weights, biases = biases)
    #mlp = MLPNetwork([len(data[0,:]),len(np.unique(labels))])
    training_data = [(data[i].reshape(-1,1),one_hot(labels[i], len(np.unique(labels))).reshape(-1,1)) for i in range(len(labels))]
    mlp.SGD(training_data = training_data,
            epochs = n,
            mini_batch_size = 20,
            eta = eta*20,
            test_data = training_data,
            )
    return [mlp.biases, mlp.weights], float(mlp.evaluate(training_data))/len(training_data)

##
# The following code has been modified 
# from http://neuralnetworksanddeeplearning.com/chap1.html
class MLPNetwork(object):
    def __init__(self, sizes, weights = None, biases = None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        if biases is None:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        else:
            self.biases = biases
        if weights is None:
            self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = weights

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpynp.arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
        


letters_dict = {'a':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,
         1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1,
        -1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,
         1, -1, -1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
        -1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,
         1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 'b':np.array([ 1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1, -1, -1,
        -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,
         1,  1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
         1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1,
         1,  1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,
        -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1]),
 'c':np.array([ 1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
         1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1]),
 'd':np.array([ 1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1,
         1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1,
        -1, -1,  1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,
        -1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1]),
 'e':np.array([ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1, -1,  1,
         1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1,  1,
         1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
 'f':np.array([ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1, -1,  1,
         1,  1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1,  1,
         1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,
         1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1]),
 'g':np.array([ 1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1,
        -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1,
        -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1]),
 'h':np.array([-1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1,
        -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
        -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1,
        -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,
         1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1]),
 'i':np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,
         1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
 'j':np.array([ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,
         1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1,
         1,  1, -1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,
        -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1]),
 'k':np.array([ 1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1, -1,
        -1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
        -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,
        -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,  1,
         1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1]),
 'l':np.array([-1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,
         1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,
         1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,
         1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1]),
 'm':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,
        -1, -1,  1,  1, -1, -1, -1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1,
        -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1, -1, -1,  1,  1,
        -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1,
        -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,
         1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 'n':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,
        -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,
        -1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1, -1,  1,  1,
        -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1, -1,
        -1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
         1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 'o':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,
        -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1,
        -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,
        -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 'p':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
         1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
         1,  1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 'q':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,
        -1, -1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1,  1, -1,
        -1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1,
        -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1]),
 'r':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
        -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,
         1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1,  1,
         1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 's':np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
        -1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1,  1,
         1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,
         1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
         1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]),
 't':np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,
         1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,
         1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,
        -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1]),
 'u':np.array([ 1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1,
        -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
         1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,
        -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1,
        -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1, -1, -1,
        -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1]),
 'v':np.array([-1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
         1, -1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1, -1, -1,
         1,  1,  1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,
         1, -1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1]),
 'w':np.array([-1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
         1, -1, -1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1,
         1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,
         1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,
        -1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1,
        -1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1]),
 'x':np.array([-1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
        -1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1, -1,
        -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,
         1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1,
         1,  1,  1, -1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1, -1,  1,  1,
         1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1]),
 'y':np.array([ 1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,
        -1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1,
        -1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,
         1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,
         1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1,
         1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1]),
 'z':np.array([ 1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,
         1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,
         1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,
         1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1,
         1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1])}
