from math import ceil, sqrt
import numpy as np
from datetime import datetime
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from .progbar import ProgressBar
import os, gzip, tarfile


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
    return int(out_height), int(out_width)


def desame(in_height, in_width, stride, kernel_size):
    '''
    calculate the input height and width from output height and width for deconvolution
    with same padding
    '''
    out_height = ceil(in_height * float(stride[0]))
    out_width = ceil(in_width * float(stride[1]))
    return int(out_height), int(out_width)


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
    return int(out_height), int(out_width)


def devalid(in_height, in_width, stride, kernel_size):
    '''
    calculate the input height and width from output height and width for deconvolution
    with valid padding
    '''
    assert isinstance(stride, (list, tuple))
    assert isinstance(kernel_size, (list, tuple))
    out_height = ceil(in_height * float(stride[0])) - 1 + kernel_size[1]
    out_width = ceil(in_width * float(stride[1])) - 1 + kernel_size[1]
    return int(out_height), int(out_width)



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
        ridx = np.random.permutation(range(len(arr))).astype(int)
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
        df = df.reindex(np.random.permutation(df.index).astype(int))

    num_train = train_valid_ratio[0] / float(sum(train_valid_ratio)) * len(df)
    num_train = int(num_train)
    df_train = df[:num_train].reset_index(drop=True)
    df_test = df[num_train:].reset_index(drop=True)
    return df_train, df_test


def ts():
    '''timestamp'''
    dt = datetime.now()
    dt = dt.strftime('%Y%m%d_%H%M_%S%f')
    return dt


def put_kernels_on_grid(kernel, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [NumKernels, Y, X, NumChannels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      num_kernels:      batchsize or number of kernels
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    kernel = tf.transpose(kernel, perm=[3, 0, 1, 2])

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (num_kernels)
    print ('grid: %d = (%d, %d)' % (num_kernels, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.shape[0] + 2 * pad
    X = kernel1.shape[1] + 2 * pad

    channels = kernel1.shape[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7


def get_file_from_url(save_path, origin, untar=False):
    datadir = os.path.dirname(save_path)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    try:
        f = open(save_path)
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

        urlretrieve(origin, save_path, dl_progress)

    if untar:
        tfile = tarfile.open(save_path, 'r:*')
        names = tfile.getnames()
        dirname = names[0]
        not_exists = [int(not os.path.exists("{}/{}".format(datadir, fname))) for fname in names]
        if sum(not_exists) > 0:
            print('Untaring file...')
            tfile.extractall(path=datadir)
        else:
            print('Files already downloaded and untarred')
        tfile.close()

    return datadir
