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
from dcll.pytorch_libdcll import *
import time

def acc(pvoutput, labels):
    return float(torch.mean((pvoutput.argmax(1) == labels.argmax(1)).float()))

if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment='MNIST Conv')

    n_epochs=200
    n_iters = 500
    in_channels = 1
    out_channels_1 = 32//2
    out_channels_2 = 48//2
    out_channels_3 = 64//2
    im_width = 28
    im_height = 28
    batch_size = 64
    output_size = 10

    layer1 = Conv2dDCLLlayer(in_channels,    out_channels = out_channels_1, im_width=im_width  , im_height=im_height  , output_size=output_size, pooling=2, padding=3, kernel_size=7).to(device)
    layer2 = Conv2dDCLLlayer(out_channels_1, out_channels = out_channels_2, im_width=im_width/2, im_height=im_height/2, output_size=output_size, pooling=2, padding=3, kernel_size=7).to(device)
    layer3 = Conv2dDCLLlayer(out_channels_2, out_channels = out_channels_3, im_width=im_width/4, im_height=im_height/4, output_size=output_size, pooling=1, padding=3, kernel_size=7).to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.SGD([layer1.i2h.weight, layer2.i2h.weight, layer3.i2h.weight] + [layer1.i2h.bias, layer2.i2h.bias, layer3.i2h.bias], lr=5e-5)

    layer1.init_hiddens(batch_size)
    layer2.init_hiddens(batch_size)
    layer3.init_hiddens(batch_size)

    avg_loss1 = 0
    avg_loss2 = 0
    avg_loss3 = 0

    from dcll.load_mnist import *
    gen_train, gen_valid, gen_test = create_data(valid=False, batch_size = batch_size)

    for epoch in range(n_epochs):
        input, labels1h = image2spiketrain(*gen_train.next())

        input = torch.Tensor(input).to(device).reshape(n_iters,
                                                       batch_size,
                                                       in_channels,
                                                       im_width,im_height)
        labels1h = torch.Tensor(labels1h).to(device)

        for iter in range(n_iters):
            if iter>150:
                optimizer.zero_grad()
                layer1.zero_grad()
                layer2.zero_grad()
                layer3.zero_grad()

            output1, pvoutput1 = layer1.forward(input[iter])
            output2, pvoutput2 = layer2.forward(output1)
            output3, pvoutput3 = layer3.forward(output2)

            if iter>150:
                losses1 = criterion(pvoutput1, labels1h[-1])
                losses2 = criterion(pvoutput2, labels1h[-1])
                losses3 = criterion(pvoutput3, labels1h[-1])

                avg_loss1 += losses1.item()
                avg_loss2 += losses2.item()
                avg_loss3 += losses3.item()

                losses1.backward()
                losses2.backward()
                losses3.backward()

                optimizer.step()

        writer.add_scalar('train/loss/layer1', avg_loss1 / n_iters, epoch)
        writer.add_scalar('train/loss/layer2', avg_loss2 / n_iters, epoch)
        writer.add_scalar('train/loss/layer3', avg_loss3 / n_iters, epoch)
        avg_loss1 = 0
        avg_loss2 = 0
        avg_loss3 = 0

        #print("Step time: {0}".format(time.time()-t0))
        print(output1.detach().cpu().numpy().mean(), output2.detach().cpu().numpy().mean(), output3.detach().cpu().numpy().mean())
        print('TRAIN Epoch {0}: Acc1 {1:1.3} Acc2 {2:1.3} Acc3 {3:1.3}'.format(epoch, acc(pvoutput1,labels1h[-1]), acc(pvoutput2, labels1h[-1]), acc(pvoutput3, labels1h[-1])))

        # input, labels1h = image2spiketrain(*gen_test.next())
        # input = torch.Tensor(input).to(device).reshape(n_iters,
        #                                                batch_size,
        #                                                in_channels,
        #                                                im_width,im_height)
        # labels1h = torch.Tensor(labels1h).to(device)

        for iter in range(n_iters-100):
            output1, pvoutput1_test = layer1.forward(input[iter])
            output2, pvoutput2_test = layer2.forward(output1)
            output3, pvoutput3_test = layer3.forward(output2)


            losses1 = criterion(pvoutput1, labels1h[-1])
            losses2 = criterion(pvoutput2, labels1h[-1])
            losses3 = criterion(pvoutput3, labels1h[-1])

            avg_loss1 += losses1.item()
            avg_loss2 += losses2.item()
            avg_loss3 += losses3.item()

        avg_loss1 /= n_iters
        avg_loss2 /= n_iters
        avg_loss3 /= n_iters

        avg_acc1 = acc(pvoutput1, labels1h[-1])
        avg_acc2 = acc(pvoutput2, labels1h[-1])
        avg_acc3 = acc(pvoutput3, labels1h[-1])

        writer.add_scalar('test/loss/layer1', avg_loss1, epoch)
        writer.add_scalar('test/loss/layer2', avg_loss2, epoch)
        writer.add_scalar('test/loss/layer3', avg_loss3, epoch)

        writer.add_scalar('test/acc/layer1', avg_acc1, epoch)
        writer.add_scalar('test/acc/layer2', avg_acc2, epoch)
        writer.add_scalar('test/acc/layer3', avg_acc3, epoch)

        print('Test Epoch {0}: L1 {1:1.3} L2 {2:1.3} L3{3:1.3} Acc1 {4:1.3} Acc2 {5:1.3} Acc3 {6:1.3}'.format(epoch, avg_loss1, avg_loss2, avg_loss3, avg_acc1, avg_acc2, avg_acc3))

        #a = np.array(states1)
        #b = np.array(states2)
    writer.close()
