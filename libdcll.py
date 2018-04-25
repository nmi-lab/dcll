#!/bin/python
#-----------------------------------------------------------------------------
# File Name : libdcll.py
# Author: Emre Neftci
#
# Creation Date : Mon 23 Apr 2018 09:18:40 AM PDT
# Last Modified : Mon 23 Apr 2018 10:03:24 PM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import tensorflow as tf
import numpy as np

def tfsigmoid(x,k=1,x0=0):
    return tf.sigmoid(k*(x-x0))

def sigmoid(x, k=1, x0=0):
    '''
    Sigmoid function. Default parameters are fitted to the spiking neuron with threshold noise=N(0,.5), alpha=1-deltat/2e-2, and tarp = 0
    '''
    y = np.empty_like(x)
    idx = x>-100
    y[~idx]=0
    y[idx] = 1. / (1 + np.exp(-k*(x[idx]-x0)))
    return y

layer_count = 0

class SNNDenseLayer(object):
    
    def __init__(self,
            layer_size,
            input_size,
            target_size,
            batch_size=50,
            lr=5e-4,
            tau=10,
            taus = 20,
            name= 'layer1'):
        """ Create a Dense SRM with Local Learning.
        
            Args:
            layer_size: An integer. The number of hidden units.
            input_size: An integer. The number of inputs per time step.
            target_size: An integer. The number of targets per time step.
        """
        self.name = name
        self.layer_size = layer_size
        if not hasattr(layer_size, '__len__'):
            self.layer_size = [self.layer_size]
        self.input_size = input_size
        self.target_size = target_size
        self.feature_size = self.input_size +self.layer_size
        self.output_factor = 1
        self.deltat=1e-3
        self.batch_size = batch_size
        self.tau = tau
        self.taus = taus
        self.alpha = 1-self.deltat/(self.tau*1e-3); #membrane time constant #6ms 
        self.alphas = 1-self.deltat/(self.taus*1e-3); #synapse time constant

        self.mod_lr = tf.placeholder(tf.float32, name = 'mod_lr')
        
        self._inputs = tf.placeholder(tf.float32, shape=[None, batch_size]+[np.prod(input_size)],
                                      name='inputs')
        self._targets = tf.placeholder(tf.float32, shape=[None, batch_size, target_size],
                                       name='targets')
        limit = np.sqrt(1e-32 / (np.prod(self.layer_size) + np.prod(self.input_size)))        
        self.initializer = tf.random_normal_initializer(0, limit)
        self.lr = lr
        self.tarp = 3

        limit = np.sqrt(6.0 / (np.prod(self.layer_size) + self.target_size))
        self.M = np.random.uniform(-limit, limit, size=[np.prod(self.layer_size)/self.output_factor,self.target_size]).astype(np.float32)
        self.bM=bM=-5

        self.is_output=False

        self.inits = [0]

    def step(self, states_prev, XY):
        """ SRM RNN step.

            Args:
            state_prev: A 7-D float32 Tensor.
            XY: A 1-D float32 Tensor with shape `[input_size+target_size]`.

            Returns:
            The updated state, with the same shape as `states_prev`.
        """
        v_prev        = tf.reshape(states_prev[0], [self.batch_size]+self.layer_size)
        isyn_prev     = tf.reshape(states_prev[1], [self.batch_size]+self.layer_size)
        dtarp_prev    = tf.reshape(states_prev[2], [self.batch_size]+self.layer_size)
        epsilon0_prev = tf.reshape(states_prev[3], [self.batch_size]+self.input_size)
        epsilon_prev  = tf.reshape(states_prev[4], [self.batch_size]+self.input_size)
        
        npsis = XY.shape[1]-self.target_size
        x = tf.reshape(XY[:, :npsis], [self.batch_size]+self.input_size)
        y = tf.reshape(XY[:, npsis:npsis+self.target_size], [self.batch_size, self.target_size])

        M = tf.convert_to_tensor(self.M)
        bM =  self.bM
        alpha = self.alpha
        alphas = self.alphas

        with tf.variable_scope('rnn_block'):
            isarp = tf.cast(dtarp_prev>0,'float32') 
            self.W_x = W_x = tf.get_variable('W_x'+self.name, shape=self.feature_size, initializer = self.initializer)
            self.b_x = b_x = tf.get_variable('b_x'+self.name, shape=self.feature_size[-1], initializer = self.initializer)
            isyn = self.alphas*isyn_prev + tf.matmul(x, W_x) + b_x
            #isyn=isyn_prev
            v = alpha*v_prev + isyn  
            #v=v_prev
            s = tf.cast(tf.sigmoid(v) > .5, 'float32') * (1-isarp)
            #s = v
            dtarp = dtarp_prev + s*self.tarp - isarp
            #dtarp = dtarp_prev
            epsilon0 = alphas*epsilon0_prev + x
            epsilon = alpha*epsilon_prev + epsilon0_prev
            pv = tf.matmul(tf.stop_gradient(epsilon), W_x) + b_x
            rhat = tf.reshape(tf.sigmoid(pv),[self.batch_size,-1])
            r = tf.sigmoid(tf.matmul(rhat,M)+bM)
            loss = tf.losses.mean_squared_error(tf.stop_gradient(y),r)

            v        = tf.reshape(v,             [self.batch_size, np.prod(self.layer_size)])                 
            isyn     = tf.reshape(isyn,          [self.batch_size, np.prod(self.layer_size)])                 
            dtarp    = tf.reshape(dtarp,         [self.batch_size, np.prod(self.layer_size)])                 
            epsilon0 = tf.reshape(epsilon0 ,     [self.batch_size, np.prod(self.input_size)])                        
            epsilon  = tf.reshape(epsilon  ,     [self.batch_size, np.prod(self.input_size)])                        
            r       = tf.reshape(r       , [self.batch_size, self.target_size])
            s_r      = s
            s        = tf.reshape(s_r,[self.batch_size, int(np.prod(self.layer_size)/self.output_factor)])
            gradW     = tf.gradients(loss,[W_x])
            gradb     = tf.gradients(loss,[b_x])
            doupdateW = tf.assign_add(self.W_x, -gradW[0]*self.lr*self.mod_lr) 
            doupdateb = tf.assign_add(self.b_x, -gradb[0]*self.lr*self.mod_lr) 
            ##Continuous output
            #output = rhat
            ##Spiking output
            output = s
        return v,isyn,dtarp,epsilon0,epsilon,r,output,doupdateW,doupdateb

    def inputs_and_targets(self):
        return tf.concat([self._inputs, self._targets], axis=2)

    def compute_predictions(self):
        """ Compute RNN states and predictions. """

        with tf.variable_scope('states'):
            sizes = [[self.batch_size]+[np.prod(self.layer_size)]]*3+\
                    [[self.batch_size]+[np.prod(self.input_size)]]*2+\
                    [[self.batch_size]+[np.prod(self.target_size)]]+\
                    [[self.batch_size]+[int(np.prod(self.layer_size)/self.output_factor)]] + [self.feature_size] + [[self.feature_size[-1]]]
            inits     = [tf.zeros(i, name='initial_state') for i in sizes]
            inits[0] += self.inits[0]
            states    = tf.scan(self.step,
                                self.inputs_and_targets(),
                                initializer=tuple(inits),
                                back_prop = False,
                                parallel_iterations = 1,
                                name='states')
        return states

    @property
    def inputs(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, input_size]`. """
        return self._inputs
    
    @property
    def targets(self):
        """ A 2-D float32 placeholder with shape `[dynamic_duration, target_size]`. """
        return self._targets
    
    @property
    def states(self):
        """ A 2-D float32 Tensor with shape `[dynamic_duration, layer_size]`. """
        return self._states
    
    @property
    def predictions(self):
        """ A 2-D float32 Tensor with shape `[dynamic_duration, target_size]`. """
        return self._predictions
    
    @property
    def loss(self):
        """ A 0-D float32 Tensor. """
        return self._loss

class SNNConvLayer(SNNDenseLayer):
    def __init__(self,
            layer_size,
            input_size,
            target_size,
            feature_size,
            pooling=2,
            batch_size = 50,
            lr=5e-4,
            tau = 10,
            taus = 20,
            name = 'layer1'):
        """ Create a convolutional SRM layer.
        
            Args:'__len__', 
            layer_size: An integer. The number of hidden units.
            input_size: An integer. The number of inputs per time step.
            target_size: An integer. The number of targets per time step.
            feature_size: Size of the convolutional kernel (kernel_height, kernel_width, channels_in, channels_out)
            pooling: dimensions of the max pooling at the output.
        """
        super(SNNConvLayer, self).__init__(
                layer_size= layer_size,
                input_size = input_size,
                target_size = target_size,
                batch_size = batch_size,
                lr = lr,
                tau = tau,
                taus = taus,
                name = name)

        self.feature_size = feature_size
        self.pooling=pooling
        self.output_factor = pooling**2
        limit = np.sqrt(6.0 / (np.prod(self.layer_size) + self.target_size))
        self.M = np.random.uniform(-limit, limit, size=[np.prod(self.layer_size)/self.output_factor,self.target_size]).astype(np.float32)
        self.bM=bM=-5

    def step(self, states_prev, XY):
        """ SRM RNN step.

        Args:
            state_prev: A 7-D float32 Tensor.
            X: A 1-D float32 Tensor with shape `[input_size+target_size]`.

        Returns:
            The updated state `h`, with the same shape as `states_prev`.
        """
        v_prev        = tf.reshape(states_prev[0], [self.batch_size]+self.layer_size)
        isyn_prev     = tf.reshape(states_prev[1], [self.batch_size]+self.layer_size)
        dtarp_prev    = tf.reshape(states_prev[2], [self.batch_size]+self.layer_size)
        epsilon0_prev = tf.reshape(states_prev[3], [self.batch_size]+self.input_size)
        epsilon_prev  = tf.reshape(states_prev[4], [self.batch_size]+self.input_size)
        
        npsis = XY.shape[1]-self.target_size
        x = tf.reshape(XY[:, :npsis], [self.batch_size]+self.input_size)
        y = tf.reshape(XY[:, npsis:npsis+self.target_size], [self.batch_size, self.target_size])

        M = tf.convert_to_tensor(self.M)
        bM =  self.bM
        alpha = self.alpha
        alphas = self.alphas

        with tf.variable_scope('rnn_block'):
            isarp = tf.cast(dtarp_prev>0,'float32') 
            self.W_x = W_x = tf.get_variable('W_x'+self.name, shape=self.feature_size, initializer = self.initializer)
            self.b_x = b_x = tf.get_variable('b_x'+self.name, shape=self.feature_size[-1], initializer = self.initializer)
            isyn = self.alphas*isyn_prev + tf.nn.conv2d(x, W_x, [1,1,1,1], padding = "SAME") + b_x
            #isyn = isyn_prev
            v = alpha*v_prev + isyn  
            #v= v_prev
            s = tf.cast(tf.sigmoid(v) > .5, 'float32') * (1-isarp)
            #s=v
            dtarp = dtarp_prev + s*self.tarp - isarp
            #dtarp = dtarp_prev
            epsilon0 = alphas*epsilon0_prev + x
            epsilon = alpha*epsilon_prev + epsilon0_prev
            pv = tf.nn.conv2d(tf.stop_gradient(epsilon), W_x, [1,1,1,1], padding = "SAME") + b_x 

            if self.pooling>1: 
                s_hat      = tf.layers.max_pooling2d(tf.sigmoid(pv),self.pooling,self.pooling)
            else:
                s_hat      = tf.sigmoid(pv)
            rhat = tf.reshape(s_hat,[self.batch_size,-1])
            r = tf.sigmoid(tf.matmul(rhat,M)+bM)
            loss = tf.losses.mean_squared_error(tf.stop_gradient(y),r)

            v        = tf.reshape(v,             [self.batch_size, np.prod(self.layer_size)])                 
            isyn     = tf.reshape(isyn,          [self.batch_size, np.prod(self.layer_size)])                 
            dtarp    = tf.reshape(dtarp,         [self.batch_size, np.prod(self.layer_size)])                 
            epsilon0 = tf.reshape(epsilon0 ,     [self.batch_size, np.prod(self.input_size)])                        
            epsilon  = tf.reshape(epsilon  ,     [self.batch_size, np.prod(self.input_size)])                        
            r       = tf.reshape(r         ,     [self.batch_size, self.target_size])

            s        = tf.reshape(tf.layers.max_pooling2d(s,self.pooling,self.pooling),[self.batch_size, -1])
            gradW     = tf.gradients(loss,[W_x])
            gradb     = tf.gradients(loss,[b_x])
            doupdateW = tf.assign_add(self.W_x, -gradW[0]*self.lr*self.mod_lr) 
            doupdateb = tf.assign_add(self.b_x, -gradb[0]*self.lr*self.mod_lr) 
            ##Continuous output
            #output = rhat
            ##Spiking output
            output = s


        return v,isyn,dtarp,epsilon0,epsilon,r,output,doupdateW,doupdateb


def DCNNConvLayer(feat_out, input_shape, ksize=5, pooling=2,  target_size=11, layer_input=None, batch_size = 64, tau=10, taus=20):
    global layer_count
    Nin = input_shape
    feat_in = input_shape[-1]
    Nhid = [input_shape[0], input_shape[1], feat_out]
    layer_count += 1
    print (input_shape)
    print (Nin)
    print (Nhid)
        
    layer = SNNConvLayer(layer_size=Nhid, input_size=Nin, feature_size=[ksize,ksize,feat_in, feat_out], target_size=target_size, batch_size = batch_size, pooling=pooling, lr = 1.0, name='layer'+str(layer_count), tau=tau, taus=taus)

    if layer_input is not None:
        layer._inputs = layer_input
    state = layer.compute_predictions()
    v,isyn,dtarp,epsilon0,epsilon,r,s,doupdateW,doupdateb = state
    return layer,state, s

def DCNNDenseLayer(feat_out, input_shape, layer_input=None, batch_size = 64, target_size=11, tau = 10, taus=20):
    import copy
    global layer_count
    Nin = input_shape
    feat_in = input_shape[-1]
    Nhid = feat_out
    layer_count += 1
    print (input_shape)
    print (Nin)
    print (Nhid)
        
    layer = SNNDenseLayer(layer_size=Nhid, input_size=Nin, target_size=target_size, batch_size = batch_size, lr = 1.0, name='layer'+str(layer_count), tau=tau, taus=taus)

    if layer_input is not None:
        layer._inputs = layer_input
    state = layer.compute_predictions()
    v,isyn,dtarp,epsilon0,epsilon,r,s,doupdateW,doupdateb = state
    return layer,state, s






































