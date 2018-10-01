#!/bin/python
#-----------------------------------------------------------------------------
# File Name : experiment_tools.py
# Author: Emre Neftci
#
# Creation Date : Thu 26 Jul 2018 09:13:34 AM PDT
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
import numpy as np

def accuracy(model, x_test, y_test):
    t = model.predict(x_test)
    return 1-(t.argmax(axis=1) == y_test.argmax(axis=1)).mean()

def annotate(d, text='', filename='notes.txt'):
    "Create a file in the Results directory, with contents text"
    f = open(d + '/' +filename, 'w')
    f.write(text)
    f.close()

def save(directory, obj=None, filename='default.pkl'):
    import pickle
    if obj == None and filename == None:
        f = open(directory + 'data.pickle', 'wb')
        pickle.dump(globaldata, f)
        f.close()
        save_source()
    elif obj == None and filename != None:
        f = open(directory + filename, 'wb')
        pickle.dump(globaldata, f)
        f.close()
    else:
        f = open(directory + filename, 'wb')
        pickle.dump(obj, f)
        f.close()
    return None

def save_source(directory):
    """
    Save all the python scripts from the current directory into the results directory
    """
    import tarfile, glob
    h = tarfile.open(directory + '/exp_scripts.tar.bz2', 'w:bz2')
    all_src = []
    all_src += glob.glob('*.py')
    for i in all_src:
        h.add(i)
    h.close()


def mksavedir(pre='Results/', exp_dir=None):
    """
    Creates a results directory in the subdirectory 'pre'. The directory name is given by ###__dd_mm_yy, where ### is the next unused 3 digit number
    """
    import os
    import tarfile
    import glob
    import time
    import os
    import fnmatch
    import warnings
    import pickle
    if pre[-1] != '/':
        pre + '/'

    if not os.path.exists(pre):
        os.makedirs(pre)
    prelist = np.sort(fnmatch.filter(os.listdir(pre), '[0-9][0-9][0-9]__*'))

    if exp_dir == None:
        if len(prelist) == 0:
            expDirN = "001"
        else:
            expDirN = "%03d" % (
                int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

        direct = time.strftime(
            pre + expDirN + "__" + "%d-%m-%Y", time.localtime())
        assert not os.path.exists(direct)

    elif isinstance(exp_dir, str):
        direct = pre + exp_dir
        if os.path.exists(direct):
            print(("Warning: overwriting directory {0}".format(direct)))
            rmtree(direct)

    else:
        raise TypeError('exp_dir should be a string')

    os.mkdir(direct)

    directory = direct + str('/')

    fh = open(
        directory + time.strftime("%H:%M:%S", time.localtime()), 'w')
    fh.close()

    print(("Created experiment directory {0}".format(directory)))
    return directory
