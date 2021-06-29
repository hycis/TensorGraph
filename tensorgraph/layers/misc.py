import tensorflow as tf
from .template import BaseLayer, BaseModel
from .merge import Sum
from ..node import StartNode, HiddenNode, EndNode
import numpy as np


class Transpose(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, perm):
        '''
        Example:
            >>> X = [3, 5, 9], Y = tf.transpose(X, perm=[0,2,1]) gives
            >>> Y = [3, 9, 5]
        '''
        self.perm = perm

    def _train_fprop(self, state_below):
        return tf.transpose(state_below, self.perm)


class Reverse(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, axis):
        '''
        Args:
            axis (list): list of axis to reverse
        '''
        self.axis = axis

    def _train_fprop(self, state_below):
        return tf.reverse(state_below, axis=self.axis)


class Flatten(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.contrib.layers.flatten(state_below)


class SetShape(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, shape):
        self.shape = shape

    def _train_fprop(self, state_below):
        shape = []
        for idx, sh in zip(range(len(state_below.get_shape())), self.shape):
            if sh <= 0:
                shape.append(state_below.shape[idx])
            else:
                shape.append(sh)
        state_below.set_shape(shape)
        return state_below


class Reshape(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, shape):
        self.shape = shape

    def _train_fprop(self, state_below):
        return tf.reshape(state_below, self.shape)


class ReduceSum(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, reduction_indices=None, keep_dims=False):
        self.reduction_indices = reduction_indices
        self.keep_dims = keep_dims

    def _train_fprop(self, state_below):
        return tf.reduce_sum(state_below, axis=self.reduction_indices,
                             keep_dims=self.keep_dims)


class ReduceMax(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, reduction_indices=None, keep_dims=False):
        self.reduction_indices = reduction_indices
        self.keep_dims = keep_dims

    def _train_fprop(self, state_below):
        return tf.reduce_max(state_below, axis=self.reduction_indices,
                             keep_dims=self.keep_dims)


class Squeeze(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, squeeze_dims=None):
        '''
        Args:
            squeeze_dims: An optional list of ints. Defaults to []. If specified,
            only squeezes the dimensions listed. The dimension index starts at 0.
            It is an error to squeeze a dimension that is not 1. Refer to tensorflow
            for details.
        '''
        self.squeeze_dims = squeeze_dims

    def _train_fprop(self, state_below):
        return tf.squeeze(state_below, self.squeeze_dims)


class Expand_Dims(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, dim):
        self.dim = dim

    def _train_fprop(self, state_below):
        return tf.expand_dims(state_below, self.dim)


class Embedding(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, cat_dim, encode_dim, embedding=None, zero_pad=False):
        """
        Args:
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
                self._W = tf.concat(axis=0, values=[zeros, self.embedding])
            else:
                self._W = self.embedding


    def _train_fprop(self, state_below):
        '''
        Args:
            state_below: is a list of indices
        '''
        return tf.gather(self._W, state_below)

    @property
    def _variables(self):
        return [self.embedding]


class Lambda(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, func, *args, **kwargs):
        '''func can be a lambda or some function that takes state_below as first arg
        '''
        self.func = func
        self.args = args
        self.kwargs = kwargs


    def _train_fprop(self, state_below):
        return self.func(state_below, *self.args, **self.kwargs)


class OneHot(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, onehot_size):
        """
        Description:
            convert indexes to onehot

        Args:
            onehot_size (int): size of dictionary for onehot

        Returns:
            given state of shape [d1, d2, ..., dk], return
            shape of [d1, d2, ..., dk, onehot_size]
        """
        self.diag = tf.diag([1.0] * onehot_size)


    def _train_fprop(self, state_below):
        '''
        Args:
            state_below: is a list of indices
        '''
        return tf.gather(self.diag, state_below)


class Slim(BaseModel):
    """
        Slim converts Tensorgraph Sequential Mode into Functional Mode.
        Similar to Keras Sequential and Functional.
        
        Example how to use Slim
        class SampleModel(BaseModel):
            @BaseModel.init_name_scope
            def __init__(self):
                self.startnode = StartNode([None])
                conv1 = Slim([self.startnode], [Conv2D(num_filters=16, kernel_size=(3,3), stride=(1,1), padding='SAME')])
                flat  = Slim([conv1.endnode],  [Flatten()])
                batch = Slim([flat.endnode],   [BatchNormalization()])
                l1    = Slim([batch.endnode],  [Linear(this_dim=30) ])
                l2    = Slim([l1.endnode],     [Linear(this_dim=10) ])
                l3    = Slim([l2.endnode],     [Linear(this_dim=5)  ])
                self.endnode = EndNode(prev=[l3.endnode])
    """
    @BaseModel.init_name_scope
    def __init__(self, start, layers, merge=Sum()):
        self.startnode = start
        out_hn         = HiddenNode(prev=start, input_merge_mode=merge, layers=layers)
        self.endnode   = EndNode(prev=[out_hn])