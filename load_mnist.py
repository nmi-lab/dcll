import keras
import numpy as np
from npamlib import spiketrains
from keras.datasets import mnist as dset
from keras.preprocessing.image import ImageDataGenerator

input_shape = [28,28,1]

def create_data(valid=False, batch_size = 100):
    (x_train, y_train), (x_test, y_test) = dset.load_data()
    if valid:
        x_test = x_train[50000:]
        y_test = y_train[50000:]
        x_train = x_train[:50000]
        y_train = y_train[:50000]
    if x_train.ndim < 4 : 
        x_train = np.expand_dims(x_train,axis=3)
    if x_test.ndim < 4 : 
        x_test = np.expand_dims(x_test,axis=3)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    nY = 10
    y_train = keras.utils.to_categorical(y_train, nY)
    y_test = keras.utils.to_categorical(y_test, nY)

    # This will do preprocessing and realtime data augmentation (None here):


    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)
    valid_dg = datagen.flow(x_test[:1000], y_test[:1000], batch_size=batch_size, shuffle=False)
    test_dg = datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
    train_dg = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
    return train_dg, valid_dg, test_dg


def image2spiketrain(x,y,min_duration=499, max_duration=500):
    batch_size = x.shape[0]
    Nout = y.shape[1]
    T = np.random.randint(min_duration,max_duration,batch_size)
    Nin = np.prod(input_shape)
    allinputs = np.zeros([batch_size,max_duration, Nin])
    for i in range(batch_size):
        st = spiketrains(T = T[i], N = Nin, rates=50*x[i].reshape(-1)).astype(np.float32)
        allinputs[i] =  np.pad(st,((0,max_duration-T[i]),(0,0)),'constant')
    allinputs = np.transpose(allinputs, (1,0,2))

    alltgt = np.zeros([max_duration, batch_size, Nout], dtype=np.float32)
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

