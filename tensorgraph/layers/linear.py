import tensorflow as tf
from template import Template


class Linear(Template):

    def __init__(self, prev_dim=None, this_dim=None, W=None, b=None):
        """
        DESCRIPTION:
            This is a fully connected layer
        PARAM:
            prev_dim (int): dimension of previous layer
            this_dim (int): dimension of this layer
            W (tensor variable): Weight of 2D tensor matrix
            b (tensor variable): bias of 2D tensor matrix
        """

        self.prev_dim = prev_dim
        self.this_dim = this_dim

        self.W = W
        if self.W is None:
            self.W = tf.Variable(tf.random_normal([self.prev_dim, self.this_dim], stddev=0.1),
                                                  name=self.__class__.__name__ + '_W')
        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.this_dim]), name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        return tf.matmul(state_below, self.W) + self.b

    @property
    def _variables(self):
        return [self.W, self.b]


class LinearMasked(Template):

    def __init__(self, prev_dim=None, this_dim=None, W=None, b=None, mask=None):
        """
        DESCRIPTION:
            This is a fully connected layer with an applied mask for partial connections
        PARAM:
            prev_dim (int): dimension of previous layer
            this_dim (int): dimension of this layer
            name (string): name of the layer
            W (tensor variable): Weight of 2D tensor matrix
            b (tensor variable): bias of 2D tensor matrix
            mask (numpy.ndarray or tensorflow placeholder): mask for partial connection
            params (list): a list of params in layer that can be updated
        """

        self.prev_dim = prev_dim
        self.this_dim = this_dim
        self.mask = mask

        self.W = W
        if self.W is None:
            self.W = tf.Variable(tf.random_normal([self.prev_dim, self.this_dim], stddev=0.1),
                                                   name=self.__class__.__name__ + '_W')
        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.this_dim]), name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        return tf.mul(tf.matmul(state_below, self.W) + self.b, self.mask)

    @property
    def _variables(self):
        return [self.W, self.b]


class SparseLinear(Template):

    def __init__(self, prev_dim=None, this_dim=None, ndim=None, W=None, b=None, batchsize=None):
        """
        DESCRIPTION:
            This is a fully connected layer with sparse inputs are two tensors
            one is index tensor of dimension [N, ndim] and another one is value
            tensor of [N]
        PARAM:
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

        self.W = W
        if self.W is None:
            self.W = tf.Variable(tf.random_normal([self.prev_dim, self.this_dim], stddev=0.1),
                                                   name=self.__class__.__name__ + '_W')
        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.this_dim]), name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        idx, val = state_below
        X = tf.SparseTensor(tf.cast(idx, 'int64'), val, shape=[self.batchsize, self.prev_dim])
        X_order = tf.sparse_reorder(X)
        XW = tf.sparse_tensor_dense_matmul(X_order, self.W, adjoint_a=False, adjoint_b=False)
        return tf.add(XW, self.b)

    @property
    def _variables(self):
        return [self.W, self.b]
