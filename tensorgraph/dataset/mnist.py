
import numpy as np
from tensorgraph.utils import make_one_hot
import struct
import numpy
import gzip

import tarfile, inspect, os
from six.moves.urllib.request import urlretrieve
from ..progbar import ProgressBar

MNIST_IMAGE_MAGIC = 2051
MNIST_LABEL_MAGIC = 2049

def get_mnist_file(fpath, origin):
    '''from David Warde-Farley'''
    datadir = os.path.dirname(fpath)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    try:
        f = open(fpath)
    except:
        print('Downloading data from',  origin)

        global progbar
        progbar = None
        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = ProgressBar(total_size)
            else:
                progbar.update(count*block_size)

        urlretrieve(origin, fpath + '.gz', dl_progress)
        progbar = None

        fin = gzip.open(fpath + '.gz', 'rb')
        fout = open(fpath, 'wb')
        fout.write(fin.read())
        fin.close()
        fout.close()

    return fpath


class open_if_filename(object):
    def __init__(self, f, mode='r', buffering=-1):
        self._f = f
        self._mode = mode
        self._buffering = buffering
        self._handle = None

    def __enter__(self):
        if isinstance(self._f, str):
            self._handle = open(self._f, self._mode, self._buffering)
        else:
            self._handle = self._f
        return self._handle

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not self._f:
            self._handle.close()


def read_mnist_images(fn, dtype=None):
    """
    '''from David Warde-Farley'''
    Read MNIST images from the original ubyte file format.

    Parameters
    ----------
    fn : str or object
        Filename/path from which to read labels, or an open file
        object for the same (will not be closed for you).

    dtype : str or object, optional
        A NumPy dtype or string that can be converted to one.
        If unspecified, images will be returned in their original
        unsigned byte format.

    Returns
    -------
    images : ndarray, shape (n_images, n_rows, n_cols)
        An image array, with individual examples indexed along the
        first axis and the image dimensions along the second and
        third axis.

    Notes
    -----
    If the dtype provided was boolean, the resulting array will
    be boolean with `True` if the corresponding pixel had a value
    greater than or equal to 128, `False otherwise.

    If the dtype provided was a float or complex dtype, the values
    will be mapped to the unit interval [0, 1], with pixel values
    that were 255 in the original unsigned byte representation
    equal to 1.0.
    """
    with open_if_filename(fn, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE_MAGIC:
            raise ValueError('wrong magic number reading MNIST image file: ' +
                             fn)
        array = numpy.fromfile(f, dtype='uint8').reshape((number, rows, cols))
    if dtype:
        dtype = numpy.dtype(dtype)
        # If the user wants booleans, threshold at half the range.
        if dtype.kind is 'b':
            array = array >= 128
        else:
            # Otherwise, just convert.
            array = array.astype(dtype)
        # I don't know why you'd ever turn MNIST into complex,
        # but just in case, check for float *or* complex dtypes.
        # Either way, map to the unit interval.
        if dtype.kind in ('f', 'c'):
            array /= 255.
    return array


def read_mnist_labels(fn):
    """
    '''from David Warde-Farley'''
    Read MNIST labels from the original ubyte file format.

    Parameters
    ----------
    fn : str or object
        Filename/path from which to read labels, or an open file
        object for the same (will not be closed for you).

    Returns
    -------
    labels : ndarray, shape (nlabels,)
        A one-dimensional unsigned byte array containing the
        labels as integers.
    """
    with open_if_filename(fn, 'rb') as f:
        magic, number = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL_MAGIC:
            raise ValueError('wrong magic number reading MNIST label file: ' +
                             fn)
        array = numpy.fromfile(f, dtype='uint8')
    return array


def Mnist(binary=True, flatten=False, onehot=True, datadir='.'):
    datadir += '/mnist/'

    url = 'http://yann.lecun.com/exdb/mnist'
    paths = []
    for fname in ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
                  't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']:
        path = get_mnist_file('{}/{}'.format(datadir,fname), origin='{}/{}.gz'.format(url,fname))
        paths.append(path)

    X_train = read_mnist_images(paths[0], dtype='float32')[:,:,:,np.newaxis]
    y_train = read_mnist_labels(paths[1])

    X_test = read_mnist_images(paths[2], dtype='float32')[:,:,:,np.newaxis]
    y_test = read_mnist_labels(paths[3])

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
        X_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))

    X = np.concatenate((X_train, X_test), axis=0)
    if binary:
        X = (X >= 0.5).astype(int)

    if onehot:
        y_train = make_one_hot(y_train, 10)
        y_test = make_one_hot(y_test, 10)

    return X_train, y_train, X_test, y_test
