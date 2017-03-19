from math import ceil
import numpy as np
from datetime import datetime

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
    if onehot_size < 450:
        dig_one = np.zeros((onehot_size, onehot_size))
        np.fill_diagonal(dig_one, 1)
        rX = dig_one[np.asarray(X)]
    else:
        # for large onehot size, this is faster
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


def split_arr(arr, train_valid_ratio=[5, 1], randomize=False, seed=None):
    assert isinstance(train_valid_ratio, (list, tuple))
    if randomize:
        print('..randomizing dataset')
        if seed:
            np.random.seed(seed)
        ridx = np.random.permutation(range(len(arr)))
        arr = arr[ridx]

    num_train = train_valid_ratio[0] / float(sum(train_valid_ratio)) * len(arr)
    num_train = int(num_train)
    return arr[:num_train], arr[num_train:]


def split_df(df, train_valid_ratio=[5, 1], randomize=False, seed=None):
    assert isinstance(train_valid_ratio, (list, tuple))
    if randomize:
        print('..randomizing dataset')
        if seed:
            np.random.seed(seed)
        df = df.reindex(np.random.permutation(df.index))

    num_train = train_valid_ratio[0] / float(sum(train_valid_ratio)) * len(df)
    num_train = int(num_train)
    return df[:num_train], df[num_train:]


def ts():
    dt = datetime.now()
    dt = dt.strftime('%Y%m%d_%H%M_%S%f')
    return dt
