from math import ceil
import numpy as np

def same(in_height, in_width, stride, kernel_size):
    '''
    description:
        describe the output dimension of an input image during pooling or convolution
    same padding:
        pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
        pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
    '''

    assert isinstance(stride, (list, tuple))
    assert isinstance(kernel_size, (list, tuple))
    out_height = ceil(float(in_height) / float(stride[0]))
    out_width  = ceil(float(in_width) / float(stride[1]))
    return out_height, out_width


def valid(in_height, in_width, stride, kernel_size):
    '''
    description:
        describe the output dimension of an input image during pooling or convolution
    valid padding:
        pad_along_height = 0
        pad_along_width = 0
    '''
    assert isinstance(stride, (list, tuple))
    assert isinstance(kernel_size, (list, tuple))
    out_height = ceil(float(in_height - kernel_size[0] + 1) / float(stride[0]))
    out_width  = ceil(float(in_width - kernel_size[1] + 1) / float(stride[1]))
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
    for i in range(len(X)):
        rX[i, X[i]] = 1

    return rX


def cat_to_num(cat, start_idx=0):
    '''
    DESCRIPTION:
        convert categorical values to numeric values
    PARAM:
        cat (list or 1d array): an array of categorical values
    RETURN:
        return numeric list and a categorical dictionary map
    '''
    cat_dict = {}
    for lbl in cat:
        if lbl not in cat_dict:
            cat_dict[lbl] = start_idx
            start_idx += 1

    nums = []
    for lbl in cat:
        nums.append(cat_dict[lbl])
    return nums, cat_dict


def cat_to_one_hot(cat):
    nums, cat_dict = cat_to_num(cat, start_idx=0)
    return make_one_hot(nums, len(cat_dict)), cat_dict
