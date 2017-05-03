
import numpy as np
from tensorgraph.utils import make_one_hot
import struct
import numpy
import gzip

import tarfile, inspect, os, sys
from six.moves.urllib.request import urlretrieve
from ..progbar import ProgressBar
from ..utils import get_file_from_url


def Cifar100(flatten=False, onehot=True, datadir='./cifar100/', fine_label=True):
    '''
    PARAM:
        fine_label: True (100 classes) False (20 classes)
    '''

    url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    save_path = '{}/cifar-100-python.tar.gz'.format(datadir)
    datadir = get_file_from_url(save_path=save_path, origin=url, untar=True)
    print('untar dir', datadir)
    sav_dir = datadir + '/cifar-100-python'
    nclass = None
    def make_data(batchnames):
        X = []
        y = []
        for data_batch in batchnames:
            fp = sav_dir + '/' + data_batch
            with open(fp, 'rb') as fin:
                # python2
                if sys.version_info.major == 2:
                    import cPickle
                    tbl = cPickle.load(fin)
                # python 3
                elif sys.version_info.major == 3:
                    import pickle
                    tbl = pickle.load(fin, encoding='bytes')

                else:
                    raise Exception('python version not 2 or 3')
                X.append(tbl[b'data'])

                if fine_label:
                    y.append(tbl[b'fine_labels'])
                    nclass = 100
                else:
                    y.append(tbl[b'coarse_labels'])
                    nclass = 20

        X = np.concatenate(X, axis=0).astype('f4')
        y = np.concatenate(y, axis=0).astype('int')
        X /= 255.0
        return X, y, nclass

    X_train, y_train, nclass = make_data(['train'])
    X_test , y_test, nclass = make_data(['test'])
    if onehot:
        y_train = make_one_hot(y_train, nclass)
        y_test = make_one_hot(y_test, nclass)

    if not flatten:
        X_train = X_train.reshape((-1, 3, 32, 32)).swapaxes(1, 3)
        X_test = X_test.reshape((-1, 3, 32, 32)).swapaxes(1, 3)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = Cifar100(flatten=False, onehot=True)
    print('X_train:', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', X_test.shape)
    print('y_test:', y_test.shape)
