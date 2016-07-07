import tensorflow as tf
import numpy as np
from template import Template



class MaxPooling(Template):
    def __init__(self, poolsize=(2, 2), stride=(1,1), border_mode='SAME'):
        '''
        DESCRIPTION:
            pooling layer
        PARAM:
            stride: two-dimensional tuple (a, b), the separation horizontally a
                or vertically b between two pools
            padding: "SAME" or "VALID"
        '''

        self.poolsize = (1,) + poolsize + (1,)
        self.stride = (1,) + stride + (1,)
        self.border_mode = border_mode

    def _train_fprop(self, state_below):
        return tf.nn.max_pool(state_below, ksize=self.poolsize,
                              strides=self.stride, padding=self.border_mode)


class Conv2D(Template):
    def __init__(self, input_channels, filters, kernel_size=(3,3), stride=(1,1),
                 W=None, b=None, border_mode='SAME'):
        '''
        PARAM:
            padding: "SAME" or "VALID"
        '''
        self.input_channels = input_channels
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.border_mode = border_mode

        self.W_shape = self.kernel_size + (self.input_channels, self.filters)
        self.W = W
        if self.W is None:
            self.W = tf.Variable(tf.random_normal(self.W_shape, stddev=0.1), name='W')

        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.random_normal([self.filters], stddev=0.1), name='b')

    def _train_fprop(self, state_below):
        '''
        state_below: (b, h, w, c)
        '''
        conv_out = tf.nn.conv2d(state_below, self.W, strides=(1,)+self.stride+(1,), padding=self.border_mode)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.W, self.b]


class Conv2D_Transpose(Template):
    def __init__(self, input_channels, filters, output_shape, stride=(1,1),
                 W=None, b=None, border_mode='SAME'):
        '''
        PARAM:
            input_channels (int)
            filters (int): output channels
            output_shape: 2D tuple of (h, w)
            padding: "SAME" or "VALID"
        '''
        self.input_channels = input_channels
        self.filters = filters
        self.output_shape = output_shape
        self.stride = stride
        self.border_mode = border_mode

        width, height = self.output_shape
        self.W_shape = (width, height, self.filters, self.input_channels)
        self.W = W
        if self.W is None:
            self.W = tf.Variable(tf.random_normal(self.W_shape, stddev=0.1), name='W')

        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.random_normal([self.filters], stddev=0.1), name='b')

    def _train_fprop(self, state_below):
        '''
        state_below: (b, h, w, c)
        '''
        batch_size = tf.shape(state_below)[0]
        deconv_shape = tf.pack((batch_size,) + self.output_shape + (self.filters,))
        conv_out = tf.nn.conv2d_transpose(state_below, self.W, output_shape=deconv_shape,
                                          strides=(1,)+self.stride+(1,), padding=self.border_mode)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.W, self.b]

# TODO
class AvgPooling(Template):
    pass
