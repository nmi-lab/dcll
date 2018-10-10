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
from dcll.pytorch_libdcll import Conv2dDCLLlayer, DenseDCLLlayer, device, DCLLClassification
from dcll.experiment_tools import mksavedir, save_source, annotate
from dcll.pytorch_utils import grad_parameters, named_grad_parameters, NetworkDumper, tonumpy
import timeit
import pickle
import numpy as np
import os

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='DCLL for MNIST')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--n_epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--no_save', type=bool, default=False, metavar='N', help='disables saving into Results directory')
    #parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--n_test_interval', type=int, default=20, metavar='N', help='how many epochs to run before testing')
    parser.add_argument('--n_test_samples', type=int, default=1500, metavar='N', help='how many test samples to use')
    parser.add_argument('--n_iters_test', type=int, default=1500, metavar='N', help='for how many ms do we present a sample during classification')
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
    return parser.parse_args()

class ReferenceConvNetwork(torch.nn.Module):
    def __init__(self, im_dims, convs, loss, opt, opt_param):
        super(ReferenceConvNetwork, self).__init__()

        def make_conv(inp, conf):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inp[0],
                                out_channels=conf[0],
                                kernel_size=conf[1],
                                padding=conf[2]),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=conf[3], stride=conf[3], padding=(conf[3]-1)//2)
            )
            layer = layer.to(device)
            return (layer, [conf[0]])

        n = im_dims
        self.layer1, n = make_conv(n, convs[0])
        self.layer2, n = make_conv(n, convs[1])
        self.layer3, n = make_conv(n, convs[2])
        def latent_size():
            with torch.no_grad():
                x = torch.zeros(im_dims).unsqueeze(0).to(device)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
            return x.shape[1:]

        # Untrained linear decoder to the predictions, just like DCLL
        self.linear = torch.nn.Linear(np.prod(latent_size()), 10).to(device)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

        self.optim = opt(self.parameters(), **opt_param)
        self.crit = loss().to(device)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.linear(x.view(x.shape[0], -1))
        return x

    def learn(self, x, labels):
        y = self.forward(x)

        self.optim.zero_grad()
        loss = self.crit(y, labels)
        loss.backward()
        self.optim.step()

    def test(self, x):
        self.y_test = self.forward(x.detach())

    def write_stats(self, writer, epoch):
        writer.add_scalar('acc/ref_net', self.acc, epoch)

    def accuracy(self, labels):
        self.acc = torch.mean((self.y_test.argmax(1) == labels.argmax(1)).float()).item()
        return self.acc


class ConvNetwork(torch.nn.Module):
    def __init__(self, im_dims, batch_size, convs,
                 target_size, act,
                 loss, opt, opt_param, lc_ampl,
                 alpha=[0.85, 0.9],
                 DCLLSlice = DCLLClassification,
                 burnin=50
    ):
        super(ConvNetwork, self).__init__()
        self.batch_size = batch_size

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

        self.dcll_slices = []
        for layer, name in zip([self.layer1, self.layer2, self.layer3],
                               ['conv1', 'conv2', 'conv3']):
            self.dcll_slices.append(
                DCLLSlice(
                    dclllayer = layer,
                    name = name,
                    batch_size = batch_size,
                    loss = loss,
                    optimizer = opt,
                    kwargs_optimizer = opt_param,
                    collect_stats = True,
                    burnin = burnin)
            )


    def learn(self, x, labels):
        spikes = x
        for i, sl in enumerate(self.dcll_slices):
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
    n_iters_test = args.n_iters_test
    im_dims = (1, 28, 28)
    target_size = 10
    # number of test samples: n_test * batch_size
    n_test = np.ceil(float(args.n_test_samples)/args.batch_size).astype(int)

    opt = torch.optim.Adamax
    opt_param = {'lr':args.lr, 'betas' : [.0, args.beta]}

    # loss = torch.nn.CrossEntropyLoss
    loss = torch.nn.SmoothL1Loss

    burnin = 50
    # format: (out_channels, kernel_size, padding, pooling)
    convs = [ (16, 7, 3, 2), (24, 7, 3, 2), (32, 7, 3, 1) ]
    # convs = [ (16, 7, 3, 2), (24, 7, 3, 2), (32, 7, 3, 1), (64, 3, 3, 1) ]

    net = ConvNetwork(im_dims, args.batch_size, convs, target_size,
                      act=torch.nn.Sigmoid(), alpha=[args.alpha, args.alphas],
                      loss=loss, opt=opt, opt_param=opt_param, lc_ampl=args.lc_ampl, burnin=burnin
    )

    ref_net = ReferenceConvNetwork(im_dims, convs, loss, opt, opt_param)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir = log_dir, comment='MNIST Conv')
    dumper = NetworkDumper(writer, net)

    if not args.no_save:
        d = mksavedir(pre=args.output)
        annotate(d, text = log_dir, filename= 'log_filename')
        annotate(d, text = str(args), filename= 'args')
        with open(os.path.join(d, 'args.pkl'), 'wb') as fp:
            pickle.dump(vars(args), fp)
        save_source(d)

    n_tests_total = np.ceil(float(args.n_epochs)/args.n_test_interval).astype(int)
    acc_test = np.empty([n_tests_total, n_test, len(net.dcll_slices)])
    acc_test_ref = np.empty([n_tests_total, n_test])

    from dcll.load_mnist import *
    gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = args.batch_size)
    all_test_data = [ gen_test.next() for i in range(n_test) ]

    for epoch in range(args.n_epochs):
        input, labels = gen_train.next()
        input_spikes, labels_spikes = image2spiketrain(input, labels,
                                                       min_duration=n_iters-1,
                                                       max_duration=n_iters)
        input_spikes = torch.Tensor(input_spikes).to(device).reshape(n_iters,
                                                                     args.batch_size,
                                                                     *im_dims)
        labels_spikes = torch.Tensor(labels_spikes).to(device)
        ref_input = torch.Tensor(input).to(device).reshape(
            args.batch_size, *im_dims
        )
        ref_label = torch.Tensor(labels).to(device)

        net.reset()
        # Train
        net.train()
        ref_net.train()
        for iter in range(n_iters):
            net.learn(x=input_spikes[iter], labels=labels_spikes[iter])
            if iter > burnin:
                ref_net.learn(x=ref_input, labels=ref_label)

        if (epoch % args.n_test_interval)==0:
            for i, test_data in enumerate(all_test_data):

                test_input, test_labels = image2spiketrain(*test_data,
                                                           min_duration=n_iters_test-1,
                                                           max_duration=n_iters_test)
                test_input = torch.Tensor(test_input).to(device).reshape(n_iters_test,
                                                                         args.batch_size,
                                                                         *im_dims)
                test_labels1h = torch.Tensor(test_labels).to(device)
                test_ref_input = torch.Tensor(test_data[0]).to(device).reshape(
                    args.batch_size, *im_dims
                )
                test_ref_label = torch.Tensor(test_data[1]).to(device)

                net.reset()
                net.eval()
                ref_net.eval()
                # Test
                for iter in range(n_iters_test):
                    net.test(x = test_input[iter])

                ref_net.test(test_ref_input)

                acc_test[epoch//args.n_test_interval, i, :] = net.accuracy(test_labels1h)
                acc_test_ref[epoch//args.n_test_interval, i] = ref_net.accuracy(test_ref_label)

                if i == 0:
                    net.write_stats(writer, epoch, comment='_batch_'+str(i))
                    ref_net.write_stats(writer, epoch)
            if not args.no_save:
                np.save(d+'/acc_test.npy', acc_test)
                np.save(d+'/acc_test_ref.npy', acc_test_ref)
                annotate(d, text = "", filename = "best result")
                parameter_dict = {
                    name: data.detach().cpu().numpy()
                    for (name, data) in net.named_parameters()
                }
                with open(d+'/parameters_{}.pkl'.format(epoch), 'wb') as f:
                    pickle.dump(parameter_dict, f)
            print("Epoch {} \t Accuracy {} \t Ref {}".format(epoch, np.mean(acc_test[epoch//args.n_test_interval], axis=0), np.mean(acc_test_ref[epoch//args.n_test_interval], axis=0)))

    writer.close()
