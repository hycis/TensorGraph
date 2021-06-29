import tensorflow as tf
from ...layers import BaseLayer


class Conv3Dx(BaseLayer):
    # refer to tensorgraph.layers.conv.Conv3D
    @BaseLayer.init_name_scope
    def __init__(self, num_filters=None, kernel_size=(3,3,3), stride=(1,1,1),
                 filters=None, b=None, padding='VALID', initializer='xavier',stddev=0.1):
        """
        3d convolution with customized initialization
        initializer options: 'xavier', 'normal'
        """
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.stddev      = stddev
        self.filter      = filters
        self.b           = b
        self.initializer = initializer

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        b,d,h,w,c = state_below.shape
        c = int(c)
        self.filter_shape = self.kernel_size + (c, self.num_filters)
        if   self.initializer == 'xavier':
            xavier = tf.contrib.layers.xavier_initializer()
            if self.filter is None:
                self.filter = tf.Variable(xavier(self.filter_shape), name=self.__class__.__name__ + '_filter')
            if self.b is None:
                self.b      = tf.Variable(xavier([self.num_filters]),name=self.__class__.__name__ + '_b')
        elif self.initializer == 'normal':
            if self.filter is None:
                self.filter = tf.Variable(tf.random_normal(self.filter_shape, stddev=self.stddev),
                                          name=self.__class__.__name__ + '_filter')
            if self.b is None:
                self.b = tf.Variable(tf.zeros([self.num_filters]), name=self.__class__.__name__ + '_b')
        else:
            raise Exception('current initializer supports only xavier, and normal')

    def _train_fprop(self, state_below):
        '''
        Args:
            state_below: (b, d, h, w, c)
        '''
        conv_out = tf.nn.conv3d(state_below, self.filter, strides=(1,)+tuple(self.stride)+(1,),
                                padding=self.padding)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.filter, self.b]

