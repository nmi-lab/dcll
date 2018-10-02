#!/usr/bin/env python
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
import torch
from dcll.pytorch_libdcll import Conv2dDCLLlayer, device, DCLLClassification
from dcll.experiment_tools import mksavedir, save_source, annotate
from dcll.pytorch_utils import grad_parameters, named_grad_parameters, NetworkDumper, tonumpy
import timeit
import pickle
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='DCLL for DVS gestures')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--n_epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no_save', type=bool, default=False, metavar='N', help='disables saving into Results directory')
    #parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--n_test_interval', type=int, default=20, metavar='N', help='how many epochs to run before testing')
    parser.add_argument('--n_test_samples', type=int, default=1500, metavar='N', help='how many test samples to use')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='N', help='learning rate (Adamax)')
    parser.add_argument('--alpha', type=float, default=.9, metavar='N', help='Time constant for neuron')
    parser.add_argument('--alphas', type=float, default=.87, metavar='N', help='Time constant for synapse')
    parser.add_argument('--beta', type=float, default=.95, metavar='N', help='Beta2 parameters for Adamax')
    parser.add_argument('--lc_ampl', type=float, default=.5, metavar='N', help='magnitude of local classifier init')
    parser.add_argument('--valid', action='store_true', default=False, help='Validation mode (only a portion of test cases will be used)')
    parser.add_argument('--comment', type=str, default='',
                        help='comment to name tensorboard files')
    parser.add_argument('--output', type=str, default='Results/',
                        help='folder name for the results')
    parser.add_argument('--skip_first', type=bool, default=False, metavar='N', help='do not train first layer')
    return parser.parse_args()

class ConvNetwork(torch.nn.Module):
    def __init__(self, im_dims, batch_size,
                 target_size, act,
                 loss, opt, opt_param, lc_ampl,
                 alpha=[0.85, 0.9], skip_first=False
    ):
        super(ConvNetwork, self).__init__()
        # format: (out_channels, kernel_size, padding, pooling)
        convs = [ (16, 7, 3, 2), (24, 7, 3, 2), (32, 7, 3, 1) ]
        self.batch_size = batch_size
        self.skip_first = skip_first

        def make_conv(inp, conf):
            layer = Conv2dDCLLlayer(in_channels = inp[0], out_channels = conf[0],
                                    kernel_size=conf[1], padding=conf[2], pooling=conf[3],
                                    im_dims=inp[1:3], # height, width
                                    target_size=target_size,
                                    alpha=alpha[0], alphas=alpha[1], act = act,
                                    lc_ampl = lc_ampl,
                                    alpharp = .65,
                                    wrp = 0,
            ).to(device).init_hiddens(batch_size)
            return layer, torch.Size([layer.out_channels]) + layer.output_shape

        n = im_dims

        self.layer1, n = make_conv(n, convs[0])
        self.layer2, n = make_conv(n, convs[1])
        self.layer3, n = make_conv(n, convs[2])

        if self.skip_first:
            # scale up the first layer weights if we don't train it
            self.layer1.i2h.weight.data.mul_(100.)

        self.dcll_slices = []
        for layer, name in zip([self.layer1, self.layer2, self.layer3],
                               ['conv1', 'conv2', 'conv3']):
            self.dcll_slices.append(
                DCLLClassification(
                    dclllayer = layer,
                    name = name,
                    batch_size = batch_size,
                    loss = loss,
                    optimizer = opt,
                    kwargs_optimizer = opt_param,
                    collect_stats = True,
                    burnin = 50)
            )


    def train(self, x, labels):
        spikes = x
        for i, sl in enumerate(self.dcll_slices):
            if self.skip_first and i==0:
                # if skip first is on we don't train the first layer
                spikes, _, pv, _ = sl.forward(spikes)
            else:
                spikes, _, pv, _ = sl.train(spikes, labels)

    def test(self, x):
        spikes = x
        for sl in self.dcll_slices:
            spikes, _, _, _ = sl.forward(spikes)

    def reset(self):
        [s.init(self.batch_size, init_states = False) for s in self.dcll_slices]
    def write_stats(self, writer, epoch, comment=""):
        [s.write_stats(writer, label = 'test'+comment+'/', epoch = epoch) for s in self.dcll_slices]

    def accuracy(self, labels):
        return [ s.accuracy(labels) for s in self.dcll_slices]


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import datetime,socket,os
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs/', 'pytorch_conv3L_mnist_', current_time + '_' + socket.gethostname() +'_' + args.comment, )
    print(log_dir)

    n_iters = 500
    n_iters_test = 2000
    im_dims = (1, 28, 28)
    target_size = 10
    # number of test samples: n_test * batch_size
    n_test = np.ceil(float(args.n_test_samples)/args.batch_size).astype(int)

    opt = torch.optim.Adamax
    opt_param = {'lr':args.lr, 'betas' : [.0, args.beta]}

    loss = torch.nn.SmoothL1Loss

    net = ConvNetwork(im_dims, args.batch_size, target_size,
                      act=torch.nn.Sigmoid(), alpha=[args.alpha, args.alphas],
                      loss=loss, opt=opt, opt_param=opt_param, lc_ampl=args.lc_ampl,
                      skip_first=args.skip_first
    )

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir = log_dir, comment='MNIST Conv')
    dumper = NetworkDumper(writer, net)

    if not args.no_save:
        d = mksavedir(pre=args.output)
        annotate(d, text = log_dir, filename= 'log_filename')
        annotate(d, text = str(args), filename= 'args.pkl')
        save_source(d)

    n_tests_total = np.ceil(float(args.n_epochs)/args.n_test_interval).astype(int)
    acc_test = np.empty([n_tests_total, n_test, len(net.dcll_slices)])

    from dcll.load_mnist import *
    gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = args.batch_size)
    all_test_data = [ gen_test.next() for i in range(n_test) ]

    for epoch in range(args.n_epochs):
        input, labels = image2spiketrain(*gen_train.next())

        input = torch.Tensor(input).to(device).reshape(n_iters,
                                                       args.batch_size,
                                                       *im_dims)

        labels1h = torch.Tensor(labels).to(device)
        net.reset()

        # Train
        for iter in range(n_iters):
            net.train(x = input[iter], labels=labels1h[-1])

        # Test
        if (epoch % args.n_test_interval)==0:
            net.reset()
            for i, test_data in enumerate(all_test_data):
                test_input, test_labels = image2spiketrain(*test_data)
                test_input = torch.Tensor(test_input).to(device).reshape(n_iters,
                                                                         args.batch_size,
                                                                         *im_dims)
                test_labels1h = torch.Tensor(test_labels).to(device)

                for iter in range(n_iters):
                    net.test(x = input[iter])

                acc_test[epoch//args.n_test_interval, i, :] = net.accuracy(labels1h)
                if i == 0:
                    net.write_stats(writer, epoch, comment='_batch_'+str(i))
            if not args.no_save:
                np.save(d+'/acc_test.npy', acc_test)
                annotate(d, text = "", filename = "best result")
                parameter_dict = {
                    name: data.detach().cpu().numpy()
                    for (name, data) in net.named_parameters()
                }
                with open(d+'/parameters_{}.pkl'.format(epoch), 'wb') as f:
                    pickle.dump(parameter_dict, f)
            print("Epoch {} \t Accuracy {}".format(epoch, acc_test[epoch//args.n_test_interval, 0, :]))



    writer.close()
