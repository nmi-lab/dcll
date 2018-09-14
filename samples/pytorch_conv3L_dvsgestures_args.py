#!/bin/python
#-----------------------------------------------------------------------------
# File Name : spikeConv2d.py
# Author: Emre Neftci
#
# Creation Date : Mon 16 Jul 2018 09:56:30 PM MDT
# Last Modified : Tue 11 Sep 2018 09:33:40 AM PDT
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from dcll.pytorch_libdcll import *
from dcll.experiment_tools import *
from dcll.load_dvsgestures_sparse import *
import argparse


parser = argparse.ArgumentParser(description='DCLL for DVS gestures')
parser.add_argument('--batchsize', type=int, default=64, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no_save', type=bool, default=False, metavar='N', help='disables saving into Results directory')
#parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)') 
parser.add_argument('--testinterval', type=int, default=20, metavar='N', help='how epochs to run before testing')
parser.add_argument('--lr', type=float, default=1e-6, metavar='N', help='learning rate (Adamax)')
parser.add_argument('--alpha', type=float, default=.9, metavar='N', help='Time constant for neuron')
parser.add_argument('--alphas', type=float, default=.87, metavar='N', help='Time constant for synapse')
parser.add_argument('--beta', type=float, default=.95, metavar='N', help='Beta2 parameters for Adamax')
parser.add_argument('--lc_ampl', type=float, default=.5, metavar='N', help='magnitude of local classifier init')
parser.add_argument('--valid', action='store_true', default=False, help='Validation mode (only a portion of test cases will be used)')

args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(0)
np.random.seed(0)




#Method for computing classification accuracy
acc = accuracy_by_vote

n_epochs = args.epochs
n_test_interval = args.testinterval
n_tests_total = n_epochs//n_test_interval+1
batch_size = args.batchsize
n_iters = 500
n_iters_test = 2000
dt = 1000 #us
in_channels = 2
ds = 4
im_dims = im_width, im_height = (128//ds, 128//ds)
out_channels_1 = 128
out_channels_2 = 256
out_channels_3 = 256
out_channels_4 = 512
out_channels_5 = 1024
out_channels_6 = 1024
target_size = 11
act=nn.Sigmoid()
#act = nn.ReLU()
#ALLConv
layer1 = Conv2dDCLLlayer(
        in_channels,
        out_channels = out_channels_1,
        im_dims = im_dims,
        target_size=target_size,
        stride=1,
        pooling=(2,2),
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = .65,
        wrp = 0,
        act = act,
        lc_ampl = args.lc_ampl).to(device)

layer2 = Conv2dDCLLlayer(
        in_channels = layer1.out_channels,
        out_channels = out_channels_2,
        im_dims = layer1.get_output_shape(),
        target_size=target_size,
        pooling=1,
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = .65,
        wrp = 0,
        act = act,
        lc_ampl = args.lc_ampl).to(device)

layer3 = Conv2dDCLLlayer(
        in_channels = layer2.out_channels,
        out_channels = out_channels_3,
        im_dims = layer2.get_output_shape(),
        target_size=target_size,
        pooling=(2,2),
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = .65,
        wrp = 0,
        act = act,
        lc_ampl = args.lc_ampl).to(device)

layer4 = Conv2dDCLLlayer(
        in_channels = layer3.out_channels,
        out_channels = out_channels_4,
        im_dims = layer3.get_output_shape(),
        target_size=target_size,
        pooling=1,
        padding=2,
        kernel_size=7,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = .65,
        wrp = 0,
        act = act,
        lc_ampl = args.lc_ampl
        ).to(device)

layer5 = Conv2dDCLLlayer(
        in_channels = layer4.out_channels,
        out_channels = out_channels_5,
        im_dims = layer4.get_output_shape(),
        target_size=target_size,
        pooling=1,
        padding=0,
        kernel_size=3,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = .65,
        wrp = 0,
        act = act,
        lc_ampl = args.lc_ampl
        ).to(device)

layer6 = Conv2dDCLLlayer(
        in_channels = layer5.out_channels,
        out_channels = out_channels_6,
        im_dims = layer5.get_output_shape(),
        target_size=target_size,
        pooling=1,
        padding=1,
        kernel_size=3,
        alpha = args.alpha,
        alphas = args.alphas,
        alpharp = .65,
        wrp = 0,
        act = act,
        lc_ampl = args.lc_ampl
        ).to(device)



#layer5 = Conv2dDCLLlayer(out_channels_4, out_channels = out_channels_5, im_width=im_width//4, im_height=im_height//4, target_size=target_size, pooling=1, padding=pad, kernel_size=ksize, alpha = .95, alphas = .9).to(device)

#AllDense-1
#layer1 = Conv2dDCLLlayer(in_channels, out_channels = out_channels_1, im_width=im_width, im_height=im_height, target_size=target_size, pooling=2, padding=pad, kernel_size=ksize, alpha = .95, alphas = .9).to(device)
#layer2 = DenseDCLLlayer(layer1.get_flat_size(), out_channels = out_channels_2, target_size=target_size, alpha = .95, alphas = .9).to(device)
#layer3 = DenseDCLLlayer(layer2.get_flat_size(), out_channels = out_channels_3, target_size=target_size, alpha = .95, alphas = .9).to(device)
#layer4 = DenseDCLLlayer(layer3.get_flat_size(), out_channels = out_channels_4, target_size=target_size, alpha = .95, alphas = .9).to(device)
#layer5 = DenseDCLLlayer(layer4.get_flat_size(), out_channels = out_channels_5, target_size=target_size, alpha = .95, alphas = .9).to(device)

#Adamax parameters { 'betas' : [.0, .99]}
opt = optim.Adamax
opt_param = {'lr':args.lr, 'betas' : [.0, args.beta]}
#opt = optim.SGD
loss = torch.nn.SmoothL1Loss
#opt_param = {'lr':3e-4}

dcll_slices = [None for i in range(6)]
dcll_slices[0] = DCLLClassification(
        dclllayer = layer1,
        name = 'conv2d_layer1',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

dcll_slices[1] = DCLLClassification(
        dclllayer = layer2,
        name = 'conv2d_layer2',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

dcll_slices[2] = DCLLClassification(
        dclllayer = layer3,
        name = 'conv2d_layer3',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

dcll_slices[3] = DCLLClassification(
        dclllayer = layer4,
        name = 'conv2d_layer4',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

dcll_slices[4] = DCLLClassification(
        dclllayer = layer5,
        name = 'conv2d_layer5',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

dcll_slices[5] = DCLLClassification(
        dclllayer = layer6,
        name = 'conv2d_layer6',
        batch_size = batch_size,
        loss = loss,
        optimizer = opt,
        kwargs_optimizer = opt_param,
        collect_stats = True,
        burnin = 50)

#Load data
gen_train, _ = create_data(
        batch_size = batch_size,
        chunk_size = n_iters,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


_, gen_test = create_data(
        batch_size = batch_size,
        chunk_size = n_iters_test,
        size = [in_channels, im_width, im_height],
        ds = ds,
        dt = dt)


acc_train = []

from tensorboardX import SummaryWriter
import datetime,socket,os
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
comment=str(args)[10:].replace(' ', '_')
log_dir = os.path.join('runs_args/', 'pytorch_conv3L_dvsgestures_args_', current_time + '_' + socket.gethostname() +'_' + comment, )
print(log_dir)

writer = SummaryWriter(log_dir = log_dir)

def generate_test(gen_test, valid=None):
    input_test, labels_test = gen_test.next()
    input_tests = []
    labels1h_tests = []
    if valid:
        n_test = 1
    else:
        n_test = int(np.ceil(input_test.shape[0]/batch_size))
    for i in range(n_test):
        input_tests.append( torch.Tensor(input_test.swapaxes(0,1))[:,i*batch_size:(i+1)*batch_size].reshape(n_iters_test,-1,in_channels,im_width,im_height))
        labels1h_tests.append(torch.Tensor(labels_test[:,i*batch_size:(i+1)*batch_size]))
    return n_test, input_tests, labels1h_tests

n_test, input_tests, labels1h_tests = generate_test(gen_test, valid = args.valid)
print('Ntest is :')
print(n_test)


[s.init(batch_size, init_states = True) for s in dcll_slices]

acc_test = np.empty([n_tests_total,n_test,len(dcll_slices)])


if __name__ == '__main__':

    if not args.no_save:
        d = mksavedir()
        annotate(d, text = log_dir, filename= 'log_filename')
        annotate(d, text = str(args), filename= 'args')
        save_source(d)
    for epoch in range(n_epochs):
        print(epoch)
        input, labels = gen_train.next()
        input = torch.Tensor(input.swapaxes(0,1)).to(device).reshape(n_iters,batch_size,in_channels,im_width,im_height)
        labels1h = torch.Tensor(labels).to(device)

        [s.init(batch_size, init_states = False) for s in dcll_slices]
        
        for iter in range(n_iters):
            output1, _, pv1 = dcll_slices[0].train(input[iter],labels1h[iter])
            output2, _, pv2 = dcll_slices[1].train(output1,    labels1h[iter])
            output3, _, pv3 = dcll_slices[2].train(output2,    labels1h[iter])
            output4, _, pv4 = dcll_slices[3].train(output3,    labels1h[iter])
            output5, _, pv5 = dcll_slices[4].train(output4,    labels1h[iter])
            output6, _, pv6 = dcll_slices[5].train(output5,    labels1h[iter])

        if (epoch%n_test_interval)==1:

            print('TEST Epoch {0}: '.format(epoch))
            for i in range(n_test):
                input_test = input_tests[i].to(device)
                labels1h_test = labels1h_tests[i]
                [s.init(batch_size, init_states = False) for s in dcll_slices]

                for iter in range(n_iters_test):
                    output1, _, _ = dcll_slices[0].forward(input_test[iter])
                    output2, _, _ = dcll_slices[1].forward(output1   )
                    output3, _, _ = dcll_slices[2].forward(output2   )
                    output4, _, _ = dcll_slices[3].forward(output3   )
                    output5, _, _ = dcll_slices[4].forward(output4   )
                    output6, _, _ = dcll_slices[5].forward(output5   )

                acc_test[epoch//n_test_interval,i,:] = [ s.accuracy(labels1h_test) for s in dcll_slices]
                acc__test_print =  ' '.join(['L{0} {1:1.3}'.format(i,v) for i,v in enumerate(acc_test[epoch//n_test_interval,i])])
                print('TEST Epoch {0} Batch {1}:'.format(epoch, i) + acc__test_print)
                [s.write_stats(writer, label = 'test/', epoch = epoch) for s in dcll_slices]

        if not args.no_save:
            np.save(d+'/acc_test.npy', acc_test)
            annotate(d, text = "", filename = "best result")

        

        #[writer.add_scalar('train/acc/layer{0}'.format(i), acc_train[-1][i], epoch) for i in range(len(dcll_slices))]
        #[writer.add_scalar('test/acc/layer{0}'.format(i), acc_test[-1][i], epoch) for i in range(len(dcll_slices))]

