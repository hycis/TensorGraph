from math import ceil, sqrt
import numpy as np
from datetime import datetime
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from .progbar import ProgressBar
import os, gzip, tarfile


def same(in_height, in_width, stride, kernel_size):
    '''
    Calculate the output dimension of an input image during pooling or convolution
    using padded with ``SAME`` mode

    Args:
        in_height (int): input height
        in_width (int): input width
        stride (tuple): tuple of (h, w)
        kernel_size (tuple): tuple of (h, w)

    Example:
        >>> pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
        >>> pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
    '''

    assert isinstance(stride, (list, tuple))
    assert isinstance(kernel_size, (list, tuple))
    out_height = same_x(in_height, stride[0], kernel_size[0])
    out_width  = same_x(in_width, stride[1], kernel_size[1])
    return int(out_height), int(out_width)


def same_x(x, stride, kernel_size):
    return ceil(float(x) / float(stride))


def same_nd(shape, stride, kernel_size):
    """
    Calculate the output dimension of an input image during pooling or convolution
    when padded with ``SAME`` mode for arbitrary n dimensional array.

    Args:
        shape (list of int): list of dimensions of n dimensional image
        stride (list of int): list of same dimension as shape
        kernel_size (list of int): list of same dimension as shape

    Returns:
        return the new dimension of the image after convolve with ``stride`` and ``kernel_size``
    """

    rshape = []
    for sh, st, sz in zip(shape, stride, kernel_size):
        rshape.append(int(same_x(sh, st, sz)))
    return rshape


def desame(in_height, in_width, stride, kernel_size):
    '''
    Calculate the input height and width from output height and width for deconvolution
    with ``SAME`` padding

    Args:
        in_height (int): input height
        in_width (int): input width
        stride (tuple): tuple of (h, w)
        kernel_size (tuple): tuple of (h, w)

    Example:
        >>> out_height = ceil(in_height * float(stride[0]))
        >>> out_width = ceil(in_width * float(stride[1]))
    '''
    out_height = desame_x(in_height, stride[0], kernel_size[0])
    out_width = desame_x(in_width, stride[1], kernel_size[1])
    return int(out_height), int(out_width)


def desame_x(x, stride, kernel_size):
    return ceil(x * float(stride))


def desame_nd(shape, stride, kernel_size):
    '''
    Calculate the input height and width from output height and width for deconvolution
    with ``SAME`` padding for arbitrary n dimensional image

    Args:
        shape (list of int): list of dimensions of n dimensional image
        stride (list of int): list of same dimension as shape
        kernel_size (list of int): list of same dimension as shape

    Example:
        >>> out_shape[0] = ceil(in_height * float(stride[0]))
        >>> out_shape[1] = ceil(in_width * float(stride[1]))
        >>> ..
    '''
    rshape = []
    for sh, st, sz in zip(shape, stride, kernel_size):
        rshape.append(int(desame_x(sh, st, sz)))
    return rshape


def valid(in_height, in_width, stride, kernel_size):
    '''
    Describe the output dimension of an input image during pooling or convolution
    using padded with ``VALID`` mode

    Args:
        in_height (int): input height
        in_width (int): input width
        stride (tuple): tuple of (h, w)
        kernel_size (tuple): tuple of (h, w)

    Example:
        >>> pad_along_height = 0
        >>> pad_along_width = 0
    '''
    assert isinstance(stride, (list, tuple))
    assert isinstance(kernel_size, (list, tuple))
    out_height = valid_x(in_height, stride[0], kernel_size[0])
    out_width = valid_x(in_width, stride[1], kernel_size[1])
    return int(out_height), int(out_width)


def valid_x(x, stride, kernel_size):
    """
    Args:
        x (int): input length
        stride (int): 1 number of stride
        kernel_size (int): 1 number of kernel_size

    Returns:
        one int number after a valid convolve
    """
    return ceil(float(x - kernel_size + 1) / float(stride))


def valid_nd(shape, stride, kernel_size):
    """
    Calculate the output dimension of an input image during pooling or convolution
    when padded with ``VALID`` mode for arbitrary n dimensional array.

    Args:
        shape (list of int): list of dimensions of n dimensional image
        stride (list of int): list of same dimension as shape
        kernel_size (list of int): list of same dimension as shape

    Returns:
        return the new dimension of the image after convolve with ``stride`` and ``kernel_size``
    """
    rshape = []
    for sh, st, sz in zip(shape, stride, kernel_size):
        rshape.append(int(valid_x(sh, st, sz)))
    return rshape

def devalid_x(x, stride, kernel_size):
    return ceil(x * float(stride)) - 1 + kernel_size


def devalid(in_height, in_width, stride, kernel_size):
    '''
    Calculate the input height and width from output height and width for deconvolution
    with ``VALID`` padding

    Args:
        in_height (int): input height
        in_width (int): input width
        stride (tuple): tuple of (h, w)
        kernel_size (tuple): tuple of (h, w)

    Example:
        >>> out_height = ceil(in_height * float(stride[0])) - 1 + kernel_size[0]
        >>> out_width = ceil(in_width * float(stride[1])) - 1 + kernel_size[1]
    '''
    assert isinstance(stride, (list, tuple))
    assert isinstance(kernel_size, (list, tuple))
    out_height = devalid_x(in_height, stride[0], kernel_size[0])
    out_width = devalid_x(in_height, stride[1], kernel_size[1])
    return int(out_height), int(out_width)


def devalid_nd(shape, stride, kernel_size):
    '''
    calculate the input height and width from output height and width for deconvolution
    with ``VALID`` padding for arbitrary n dimensional image

    Args:
        shape (list of int): list of dimensions of n dimensional image
        stride (list of int): list of same dimension as shape
        kernel_size (list of int): list of same dimension as shape

    Example:
        >>> out_shape[0] = ceil(in_height * float(stride[0])) - 1 + kernel_size[0]
        >>> out_shape[1] = ceil(in_width * float(stride[1])) - 1 + kernel_size[1]
    '''

    rshape = []
    for sh, st, sz in zip(shape, stride, kernel_size):
        rshape.append(int(devalid_x(sh, st, sz)))
    return rshape


def make_one_hot(X, onehot_size):
    """
    Make a one-hot version of X

    Args:
        X: 1d numpy with each value in X representing the class of X
        onehot_size: length of the one hot vector

    Returns:
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
    Convert categorical values to numeric values

    Args:
        cat (list or 1d array): an array of categorical values

    Returns:
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
    assert isinstance(train_valid_ratio, (list, tuple)) and len(train_valid_ratio) == 2
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
    assert isinstance(train_valid_ratio, (list, tuple)) and len(train_valid_ratio) == 2
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


def put_kernels_on_grid(kernel, num_kernels, pad = 1):
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


def shuffle(arr):
    shf_idx = np.arange(len(arr))
    np.random.shuffle(shf_idx)
    return arr[shf_idx]


class MakeTFRecords(object):
    '''general framework for creating and reading tfrecords
    '''

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    @staticmethod
    def make_tfrecords_from_arrs(data_records, save_path, dtype=np.float32):
        '''
        Args:
            data_records: dictionary of arrs of format {'X': X_arr, 'y': y_arr, ...}
                where first dimension of the array is the example number.
        '''
        first = True
        for k in data_records:
            if first:
                arr_size = len(data_records[k])
                first = False
            else:
                assert len(data_records[k]) == arr_size, 'not all array in data_records is of same size'

        names = []
        arr_grps = []
        for name, arr_grp in data_records.items():
            arr_grp = arr_grp.astype(dtype)
            names.append(name)
            arr_grps.append(arr_grp)

        writer = tf.python_io.TFRecordWriter(save_path)
        for record in zip(*arr_grps):
            feature = {}
            for name, arr in zip(names, record):
                arr_str = arr.tostring()
                feature[name] = MakeTFRecords._bytes_feature(arr_str)
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


    @staticmethod
    def read_arrs_from_tfrecords(tfrecords_filename, data_shapes, dtype=np.float32):
        '''
        Args:
            data_shapes (dict): dictionary of the record name and shape example:
                {'X':[32,32], 'y':[10]} where the shape is the shape per image
        '''
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
        arrs = []
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            record = []
            for name, shape in data_shapes.items():
                data_raw = example.features.feature[name].bytes_list.value[0]
                data_arr = np.frombuffer(data_raw, dtype=dtype)
                data_arr = data_arr.reshape(shape)
                record.append(data_arr)
            arrs.append(record)
        return arrs


    @staticmethod
    def read_and_decode(tfrecords_filename_list, data_shapes, batch_size, dtype=tf.float32,
                        capacity=None, min_after_dequeue=None):
        '''
        Args:
            tfrecords_filename_list (list): list of tfrecords paths
            data_shapes (dict): dictionary of the record name and shape example:
                {'X':[32,32], 'y':[10]} where the shape is the shape per image
        '''
        filename_queue = tf.train.string_input_producer(tfrecords_filename_list)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = {}
        for name in data_shapes:
            features[name] = tf.FixedLenFeature([], tf.string)

        features = tf.parse_single_example(serialized_example, features=features)

        records = []
        names = []
        for name, arr_raw in features.items():
            data = tf.decode_raw(arr_raw, dtype)
            data = tf.reshape(data, data_shapes[name])
            records.append(data)
            names.append(name)
        capacity = capacity if capacity else 10*batch_size
        min_after_dequeue = min_after_dequeue if min_after_dequeue else 5*batch_size
        batch_records = tf.train.shuffle_batch(records, batch_size=batch_size,
                                               capacity=capacity,
                                               num_threads=4,
                                               min_after_dequeue=min_after_dequeue)
        if not isinstance(batch_records, (list, tuple)):
            batch_records = [batch_records]
        return zip(names, batch_records)


class MakeTFRecords_tfdata(object):
    '''general framework for creating and reading tfrecords using tf.data API
    '''

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _read_and_decode(example_proto,data_shape,dtypes):
        """
        Read and decode tfrecord to tensors

        Args:
            example_proto: A scalar string Tensor, a single serialized Example.
            data_shape: dictionary of data shape
            dtypes: list of dtypes in tf.

        Returns:
            element: dictionary of data.
        """
        features = {}
        for name in data_shape:
            features[name] = tf.FixedLenFeature([], tf.string)
        parsed_features = tf.parse_single_example(example_proto, features)
        count = 0
        res = {}
        for name in data_shape:
            res[name] = parsed_features[name]
            if dtypes[count]!=str:
                res[name]=tf.decode_raw(res[name],dtypes[count])
            if dtypes[count]==tf.float32 or dtypes[count]==tf.float64:
                res[name]=tf.convert_to_tensor(res[name],dtype=dtypes[count])
                if data_shape[name]:
                    res[name]=tf.reshape(res[name],shape=data_shape[name])
            count += 1
        return res


    @staticmethod
    def make_tfrecords_from_arrs(data_records,save_path,dtypes=[np.float32,np.float32]):
        """
        Make tfrecords from numpy array (and list of string (patientnames etc.))

        Args:
            data_records: dictionary. contains data to be made into tfrecords. Can be numpy array.
            save_path: string. Path to save tfrecords.
            dtypes: list of dtype in numpy. Should have same length of data_records.

        Returns:
            None
        """
        assert len(dtypes)==len(data_records),"length of dtypes should be same length of data_records"
        writer=tf.python_io.TFRecordWriter(save_path)
        first = True
        for k in data_records:
            if first:
                arr_size = len(data_records[k])
                first = False
            else:
                assert len(data_records[k]) == arr_size, 'not all components in data_records is of same size'
        names = []
        arr_grps = []
        count = 0
        for name, arr_grp in data_records.items():
            if type(arr_grp)==np.ndarray:
                arr_grp = arr_grp.astype(dtypes[count])
            names.append(name)
            arr_grps.append(arr_grp)
            count += 1
        for record in zip(*arr_grps):
            feature = {}
            for name, arr in zip(names, record):
                if type(arr)==np.ndarray or type(arr)==np.uint8:
                    arr_str = arr.tostring()
                    feature[name] = MakeTFRecords_tfdata._bytes_feature(arr_str)
                elif type(arr)==str:
                    arr = np.asarray(arr).tostring()
                    feature[name] = MakeTFRecords_tfdata._bytes_feature(arr)

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def read_tensor_from_tfrecords(n_exp,tfrecords_filenames,data_shape,dtypes=[tf.float32,tf.float32],batch_size=8,num_epochs=1,phase='training'):
        """
        Read tensors from numpy array (and list of string (patientnames etc.))

        Args:
            n_exp: total number of sample of all the tfrecords_filenames
            tfrecords_filenames: list of tfrecords files
            data_shape: dictionary of data shape
            dtypes: list of dtypes in tf.
            batch_size: integer. batch size.
            num_epochs: integer. number of epochs.
            phase: string. Either "training" or "validation". To determine whether shuffle.

        Returns:
            trainiterator: an iterator. To be initialize later!
            element: dictionary of data.
        """
        assert phase=="training" or phase=="validation", "phase must be <training> or <validation>"
        with tf.device('/cpu:0'):
            with tf.name_scope(phase) as scope:
                traindataset = tf.data.TFRecordDataset(tfrecords_filenames)
                traindataset = traindataset.map(map_func=lambda x:MakeTFRecords_tfdata._read_and_decode(x,data_shape,dtypes),num_parallel_calls=4)
                if phase=='training':
                    traindataset = traindataset.shuffle(buffer_size=n_exp)
                traindataset = traindataset.prefetch(buffer_size=n_exp)
                traindataset = traindataset.batch(batch_size)
                traindataset = traindataset.repeat(num_epochs)
                trainiterator = traindataset.make_initializable_iterator()
                element = trainiterator.get_next()
        return trainiterator,element
