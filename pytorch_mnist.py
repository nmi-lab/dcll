#!/bin/python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import time
import os
from pytorch_libdcll import *
from load_mnist import *
import pyNSATlib as nsat
import matplotlib as plt

# Call out to nsat
# nsat = True
nsat = False


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


def run_nsat(config_file):
    # Call the C NSAT
    print("Running C NSAT!")
    print('End %s:run_nsat()' % (os.path.splitext(os.path.basename(__file__))[0]))
    nsat.run_c_nsat(config_file)
    cfg = nsat.ConfigurationNSAT()    

    # Load the results (read binary files)
    c_nsat_reader = nsat.C_NSATReader(cfg, config_file)
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


def acc(pvoutput, labels):
    return float(torch.mean((pvoutput.argmax(1) == labels[-1].argmax(1)).float()))


def pytorch_mnist_benchmark():
    n_epochs=200
    n_iters = 500
    in_channels = 1
    out_channels_1 = 100
    out_channels_2 = 100
    im_width = 28
    im_height = 28
    batch_size = 32

    if ( not nsat ):
        layer1 = DenseDCLLlayer(im_width*im_height, out_channels = out_channels_1).to(device)
        layer2 = DenseDCLLlayer(out_channels_1, out_channels = out_channels_2).to(device)
    else:
        layer1, layer2 = setup_nsat()

    gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = batch_size)
    
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer1 = optim.SGD([layer1.i2h.weight, layer1.i2h.bias], lr=1e-5)
    optimizer2 = optim.SGD([layer2.i2h.weight, layer2.i2h.bias], lr=1e-5)

    for epoch in range(n_epochs):
        input, labels1h = image2spiketrain(*gen_train.next())
        input = torch.Tensor(input).to(device)
        labels1h = torch.Tensor(labels1h).to(device)
        states1 = []
        states2 = []

        isyn1, vmem1, eps01, eps11 = layer1.init_hiddens(batch_size)
        isyn2, vmem2, eps02, eps12 = layer2.init_hiddens(batch_size)
        
        for iter in range(n_iters):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            layer1.zero_grad()
            layer2.zero_grad()
            isyn1 = isyn1.detach()
            vmem1 = vmem1.detach()
            eps01 = eps01.detach()
            eps11 = eps11.detach()
            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)

            isyn2 = isyn2.detach()
            vmem2 = vmem2.detach()
            eps02 = eps02.detach()
            eps12 = eps12.detach()
            output1 = output1.detach()

            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)

            states1.append(np.array(output1.detach().cpu().numpy()))
            states2.append(np.array(output2.detach().cpu().numpy()))
            if iter>50:
                losses1 = criterion1(pvoutput1, labels1h[-1])
                losses2 = criterion2(pvoutput2, labels1h[-1])

                losses1.backward()
                losses2.backward()
                optimizer1.step()
                optimizer2.step()
        #print('Epoch {0}: L1 {1:1.3}  L2 {2:1.3} Acc1 {3:1.3} Acc2 {4:1.3}'.format(epoch, losses1.cpu(), losses2.cpu(), acc(pvoutput1,labels1h), acc(pvoutput2, labels1h)))
        a = np.array(states1)
        b = np.array(states2)

        input, labels1h = image2spiketrain(*gen_train.next())
        input = torch.Tensor(input).to(device)
        labels1h = torch.Tensor(labels1h).to(device)

        for iter in range(n_iters):
            isyn1, vmem1, eps01, eps11, output1, pvoutput1 = layer1.forward(input[iter], isyn1, vmem1, eps01, eps11)
            isyn2, vmem2, eps02, eps12, output2, pvoutput2 = layer2.forward(output1, isyn2, vmem2, eps02, eps12)

        print('Test Epoch {0}: L1 {1:1.3}  L2 {2:1.3} Acc1 {3:1.3} Acc2 {4:1.3}'.format(epoch, losses1.cpu(), losses2.cpu(), acc(pvoutput1,labels1h), acc(pvoutput2, labels1h)))
        a = np.array(states1)
        b = np.array(states2)


if __name__ == '__main__':
    print('Begin %s:main()' % (os.path.splitext(os.path.basename(__file__))[0]))
    start_t = time.perf_counter()
    
    pytorch_mnist_benchmark()
    
    print("End %s:main() , running time: %f seconds" % (os.path.splitext(os.path.basename(__file__))[0], time.perf_counter()-start_t))
    