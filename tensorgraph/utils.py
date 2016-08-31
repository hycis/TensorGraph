from math import ceil
import numpy as np

def same(in_height, in_width, strides, filters):
    '''
    description:
        describe the output dimension of an input image during pooling or convolution
    same padding:
        pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
        pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
    '''

    assert isinstance(strides, (list, tuple))
    assert isinstance(filters, (list, tuple))
    out_height = ceil(float(in_height) / float(strides[0]))
    out_width  = ceil(float(in_width) / float(strides[1]))
    return out_height, out_width


def valid(in_height, in_width, strides, filters):
    '''
    description:
        describe the output dimension of an input image during pooling or convolution
    valid padding:
        pad_along_height = 0
        pad_along_width = 0
    '''
    assert isinstance(strides, (list, tuple))
    assert isinstance(filters, (list, tuple))
    out_height = ceil(float(in_height - filters[0] + 1) / float(strides[0]))
    out_width  = ceil(float(in_width - filters[1] + 1) / float(strides[1]))
    return out_height, out_width


def make_one_hot(X, onehot_size):
    """
    DESCRIPTION:
        Make a one-hot version of X
    PARAM:
        X: 1d numpy with each value in X representing the class of X
        onehot_size: length of the one hot vector
    RETURN:
        2d numpy tensor, with each row been the onehot vector
    """

    rX = np.zeros((len(X), onehot_size))
    for i in xrange(len(X)):
        rX[i, X[i]] = 1

    return rX
