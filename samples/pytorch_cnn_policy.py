#!/usr/bin/env python
#-----------------------------------------------------------------------------
# File Name : pytorch_cnn_policy.py
# Author: Jacques Kaiser
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
from dcll.pytorch_libdcll import *
import time
import numpy as np
import traceback, sys, code

class LayerUpdater(object):
    def __init__(self, layer, batch_size):
        self.layer = layer
        self.isyn, self.vmem, self.eps0, self.eps1 = layer.init_hiddens(batch_size)
    def __call__(self, input):
        self.isyn, self.vmem, self.eps0, self.eps1, output, pvoutput = self.layer.forward(input, self.isyn, self.vmem, self.eps0, self.eps1)
        return output, pvoutput



class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, batch_size):
        self.batch_size = batch_size

        layer1 = Conv2dDCLLlayer(in_channels = ob_space.shape[2],
                                 out_channels = 16,
                                 im_width= ob_space.shape[1],
                                 im_height= ob_space.shape[0],
                                 # error signal dimension
                                 output_size=ac_space.shape[0],
                                 pooling=2,
                                 padding=3,
                                 kernel_size=7).to(device)

        layer2 = Conv2dDCLLlayer(in_channels = 16,
                                 out_channels = 32,
                                 im_width= ob_space.shape[1] // 2,
                                 im_height= ob_space.shape[0] // 2,
                                 # error signal dimension
                                 output_size=ac_space.shape[0],
                                 pooling=2,
                                 padding=3,
                                 kernel_size=7).to(device)

        layer3 = DenseDCLLlayer(in_channels = 32 * 7 * 7,
                                out_channels = ob_space.shape[0],
                                output_size=ac_space.shape[0]).to(device)
        # layer3 = DenseDCLLlayer(in_channels = 32,
        #                         out_channels = 256).to(device)

        self.layers = [layer1, layer2, layer3]
        self.layer_updaters = [ LayerUpdater(l, batch_size) for l in self.layers ]

    def forward(self, input):
        output = input
        # return array of tuples (output and pvoutput) for all layers
        ret = []
        for l_up in self.layer_updaters[:2]:
            output, pvout = l_up(output)
            ret.append([output, pvout])

        # flatten input for dense layer
        output, pvout = self.layer_updaters[2](output.reshape(self.batch_size,
                                                              self.layers[2].in_channels
        ))
        ret.append([output, pvout])

        return np.array(ret)

    def get_parameters(self):
        weights = [ l.i2h.weight for l in self.layers]
        bias = [ l.i2h.bias for l in self.layers]
        return weights + bias

    def zero_grad(self):
        [ l.zero_grad() for l in self.layers ]

def main():
    from gym.spaces.box import Box
    ob_space = Box(low=-1.0, high=1.0, shape=(28,28,1))
    ac_space = Box(low=-1.0, high=1.0, shape=(5,))
    batch_size = 64
    policy = CnnPolicy(
        ob_space=ob_space,
        ac_space=ac_space,
        batch_size = batch_size
    )

    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD(policy.get_parameters(), lr=5e-5)

    # random spiketrain
    n_samples = 200
    # roll observation space to put in_channel in front (instead of in the back for gym)
    input = (torch.rand(n_samples, batch_size, *np.roll(ob_space.shape, 1)) > 0.7).float().to(device)
    labels = torch.ones(batch_size, *ac_space.shape).to(device)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

    for i_batch, batch in enumerate(input):
        policy.zero_grad()
        optimizer.zero_grad()
        out = policy.forward(batch)
        pvout = out[:,1]

        losses_per_layer = [ criterion(pv, label) for pv, label in zip(pvout,
                                                                       # repeat labels for each layer
                                                                       [ labels for i in policy.layers ]) ]

        [ loss.backward() for loss in losses_per_layer ]

        writer.add_scalar('loss/layer1', losses_per_layer[0], i_batch)
        writer.add_scalar('loss/layer2', losses_per_layer[1], i_batch)
        writer.add_scalar('loss/layer3', losses_per_layer[2], i_batch)

        writer.add_image('io/input', batch, i_batch)
        writer.add_image('io/output', out[2,0], i_batch)
        writer.add_image('io/label', labels, i_batch)


        print( [o.detach().cpu().numpy().mean() for o in out[:,0]] )
        print('TRAIN batch {0} loss1 {1:1.3} loss2 {2:1.3} loss3 {3:1.3}'.format(
            i_batch, *losses_per_layer))

        optimizer.step()

    writer.close()

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        last_frame = lambda tb=tb: last_frame(tb.tb_next) if tb.tb_next else tb
        frame = last_frame().tb_frame
        ns = dict(frame.f_globals)
        ns.update(frame.f_locals)
        code.interact(local=ns)
