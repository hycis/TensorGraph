import tensorflow as tf
from .template import BaseLayer


class Linear(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, this_dim=None, W=None, b=None, stddev=0.1):
        """
        Description:
            This is a fully connected layer

        Args:
            this_dim (int): dimension of this layer
        """
        self.this_dim = this_dim
        self.stddev = stddev
        self.W = W
        self.b = b


    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        prev_dim = int(state_below.shape[1])
        if self.W is None:
            self.W = tf.Variable(tf.random_normal([prev_dim, self.this_dim], stddev=self.stddev),
                                                  name=self.__class__.__name__ + '_W')
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.this_dim]), name=self.__class__.__name__ + '_b')


    def _train_fprop(self, state_below):
        return tf.matmul(state_below, self.W) + self.b


    @property
    def _variables(self):
        return [self.W, self.b]


class LinearMasked(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, this_dim=None, W=None, b=None, mask=None, stddev=0.1):
        """
        Description:
            This is a fully connected layer with an applied mask for partial connections

        Args:
            this_dim (int): dimension of this layer
            name (string): name of the layer
            W (tensor variable): Weight of 2D tensor matrix
            b (tensor variable): bias of 2D tensor matrix
            mask (numpy.ndarray or tensorflow placeholder): mask for partial connection
            params (list): a list of params in layer that can be updated
        """

        self.this_dim = this_dim
        self.mask = mask
        self.stddev = stddev
        self.W = W
        self.b = b


    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        prev_dim = int(state_below.shape[1])
        if self.W is None:
            self.W = tf.Variable(tf.random_normal([prev_dim, self.this_dim], stddev=self.stddev),
                                                   name=self.__class__.__name__ + '_W')
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.this_dim]), name=self.__class__.__name__ + '_b')


    def _train_fprop(self, state_below):
        return tf.multiply(tf.matmul(state_below, self.W) + self.b, self.mask)


    @property
    def _variables(self):
        return [self.W, self.b]


class SparseLinear(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, prev_dim=None, this_dim=None, W=None, b=None, batchsize=None, stddev=0.1):
        """
        Description:
            This is a fully connected layer with sparse inputs are two tensors
            one is index tensor of dimension [N, prev_dim] and another one is value
            tensor of [N]

        Args:
            prev_dim (int): dimension of previous layer
            this_dim (int): dimension of this layer
            name (string): name of the layer
            W (tensor variable): Weight of 2D tensor matrix
            b (tensor variable): bias of 2D tensor matrix
            params (list): a list of params in layer that can be updated
        """

        self.prev_dim = prev_dim
        self.this_dim = this_dim
        self.batchsize = batchsize
        self.stddev = stddev
        self.W = W
        self.b = b

        if self.W is None:
            self.W = tf.Variable(tf.random_normal([self.prev_dim, self.this_dim], stddev=self.stddev),
                                                   name=self.__class__.__name__ + '_W')
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.this_dim]), name=self.__class__.__name__ + '_b')


    def _train_fprop(self, state_below):
        idx, val = state_below
        X = tf.SparseTensor(tf.cast(idx, 'int64'), val, dense_shape=[self.batchsize, self.prev_dim])
        X_order = tf.sparse_reorder(X)
        XW = tf.sparse_tensor_dense_matmul(X_order, self.W, adjoint_a=False, adjoint_b=False)
        return tf.add(XW, self.b)


    @property
    def _variables(self):
        return [self.W, self.b]
