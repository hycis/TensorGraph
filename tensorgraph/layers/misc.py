import tensorflow as tf
from template import Template
import numpy as np


class Flatten(Template):

    def _train_fprop(self, state_below):
        shape = state_below.get_shape().as_list()
        return tf.reshape(state_below, [-1, np.prod(shape[1:])])


class Reshape(Template):

    def __init__(self, shape):
        self.shape = shape

    def _train_fprop(self, state_below):
        return tf.reshape(state_below, self.shape)


class ReduceSum(Template):

    def __init__(self, reduction_indices=None, keep_dims=False):
        self.reduction_indices = reduction_indices
        self.keep_dims = keep_dims

    def _train_fprop(self, state_below):
        return tf.reduce_sum(state_below, reduction_indices=self.reduction_indices,
                             keep_dims=self.keep_dims)


class Squeeze(Template):

    def __init__(self, squeeze_dims=None):
        '''
        PARAM:
            squeeze_dims: An optional list of ints. Defaults to []. If specified,
            only squeezes the dimensions listed. The dimension index starts at 0.
            It is an error to squeeze a dimension that is not 1. Refer to tensorflow
            for details.
        '''
        self.squeeze_dims = squeeze_dims

    def _train_fprop(self, state_below):
        return tf.squeeze(state_below, self.squeeze_dims)


class Embedding(Template):
    def __init__(self, cat_dim, encode_dim, embedding=None, zero_pad=False):
        """
        DESCRIPTION:
            embedding
        PARAM:
            cat_dim (int): number of categories
            encode_dim (int): dense encoding of the categories
            embedding (tensor variable): embedding of 2D tensor variable matrix
            zero_pad (bool): whether should initialize zero embedding for sequence
                with zero paddings, zero pad is added to the first row of the embedding,
                and will not be updated by back-propagation.
        """

        self.cat_dim = cat_dim
        self.encode_dim = encode_dim
        self._W = self.embedding = embedding

        if self._W is None:
            embed = tf.random_uniform([self.cat_dim, self.encode_dim], minval=-1, maxval=1)
            self.embedding = tf.Variable(embed, name=self.__class__.__name__ + '_embedding')
            if zero_pad:
                zeros = tf.zeros([1, self.encode_dim])
                self._W = tf.concat(0, [zeros, self.embedding])
            else:
                self._W = self.embedding


    def _train_fprop(self, state_below):
        '''state_below is a list of indices
        '''
        return tf.gather(self._W, state_below)

    @property
    def _variables(self):
        return [self.embedding]
