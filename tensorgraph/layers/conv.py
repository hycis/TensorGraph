import tensorflow as tf
import numpy as np
from template import Template



class MaxPooling(Template):
    def __init__(self, poolsize=(2, 2), stride=(1,1), padding='VALID'):
        '''
        DESCRIPTION:
            pooling layer
        PARAM:
            stride: two-dimensional tuple (a, b), the separation horizontally a
                or vertically b between two pools
            padding: "SAME" pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
                            pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
                      or
                     "VALID" padding is always 0
        '''

        self.poolsize = (1,) + poolsize + (1,)
        self.stride = (1,) + stride + (1,)
        self.padding = padding

    def _train_fprop(self, state_below):
        return tf.nn.max_pool(state_below, ksize=self.poolsize,
                              strides=self.stride, padding=self.padding,
                              data_format='NHWC', name=None)


class AvgPooling(Template):
    def __init__(self, poolsize=(2, 2), stride=(1,1), padding='VALID'):
        '''
        DESCRIPTION:
            pooling layer
        PARAM:
            stride: two-dimensional tuple (a, b), the separation horizontally a
                or vertically b between two pools
            padding: "SAME" pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
                            pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
                      or
                     "VALID" padding is always 0
        '''

        self.poolsize = (1,) + poolsize + (1,)
        self.stride = (1,) + stride + (1,)
        self.padding = padding

    def _train_fprop(self, state_below):
        return tf.nn.avg_pool(state_below, ksize=self.poolsize,
                              strides=self.stride, padding=self.padding,
                              data_format='NHWC', name=None)


class Conv2D(Template):
    def __init__(self, input_channels, num_filters, kernel_size=(3,3), stride=(1,1),
                 filter=None, b=None, padding='VALID'):
        '''
        PARAM:
            padding: "SAME" pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
                            pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
                      or
                     "VALID" padding is always 0
        '''
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.filter_shape = self.kernel_size + (self.input_channels, self.num_filters)
        self.filter = filter
        if self.filter is None:
            self.filter = tf.Variable(tf.random_normal(self.filter_shape, stddev=0.1),
                                      name=self.__class__.__name__ + '_filter')

        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.num_filters]), name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        '''
        state_below: (b, h, w, c)
        '''
        conv_out = tf.nn.conv2d(state_below, self.filter, strides=(1,)+self.stride+(1,),
                                padding=self.padding, data_format='NHWC')
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.filter, self.b]


class Conv2D_Transpose(Template):
    def __init__(self, input_channels, num_filters, output_shape, kernel_size=(3,3), stride=(1,1),
                 filter=None, b=None, padding='VALID'):
        '''
        PARAM:
            input_channels (int)
            num_filters (int): output channels
            output_shape: 2D tuple of (h, w)
            padding: "SAME" pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
                            pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
                      or
                     "VALID" padding is always 0
        '''
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.output_shape = output_shape
        self.stride = stride
        self.padding = padding

        height, width = self.output_shape
        self.filter_shape = kernel_size + (self.num_filters, self.input_channels)
        self.filter = filter
        if self.filter is None:
            self.filter = tf.Variable(tf.random_normal(self.filter_shape, stddev=0.1),
                                      name=self.__class__.__name__ + '_filter')

        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.num_filters]), name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        '''
        state_below: (b, h, w, c)
        '''
        batch_size = tf.shape(state_below)[0]
        width, height = self.output_shape
        deconv_shape = tf.pack((batch_size, int(height), int(width), self.num_filters))
        conv_out = tf.nn.conv2d_transpose(value=state_below, filter=self.filter, output_shape=deconv_shape,
                                          strides=(1,)+self.stride+(1,), padding=self.padding)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.filter, self.b]


class Conv3D(Template):
    def __init__(self, input_channels, num_filters, kernel_size=(3,3,3), stride=(1,1,1),
                 filter=None, b=None, padding='VALID'):
        '''
        PARAM:
            kernel_size: [filter_depth, filter_height, filter_width]
            stride: [stride_depth, stride_height, stride_width]
            padding: "SAME" pad_along_depth = ((out_depth - 1) * stride[0] + filter_depth - in_depth)
                            pad_along_height = ((out_height - 1) * stride[1] + filter_height - in_height)
                            pad_along_width = ((out_width - 1) * stride[2] + filter_width - in_width)

                      or
                     "VALID" padding is always 0
        '''
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.filter_shape = self.kernel_size + (self.input_channels, self.num_filters)
        self.filter = filter
        if self.filter is None:
            self.filter = tf.Variable(tf.random_normal(self.filter_shape, stddev=0.1),
                                      name=self.__class__.__name__ + '_filter')

        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.num_filters]), name=self.__class__.__name__ + '_b')


    def _train_fprop(self, state_below):
        '''
        state_below: (b, d, h, w, c)
        '''
        conv_out = tf.nn.conv3d(state_below, self.filter, strides=(1,)+self.stride+(1,),
                                padding=self.padding)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.filter, self.b]
