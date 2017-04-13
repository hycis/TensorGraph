import tensorflow as tf
from .template import Template
from ..utils import ts

# TODO
class L2_Normalize(Template):
    pass


class BatchNormalization(Template):

    def __init__(self, dim, layer_type, short_memory=0.01):
        '''
        REFERENCE:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        PARAMS:
            short_memory: short term memory
                y_t is the latest value, the moving average x_tp1 is calculated as
                x_tp1 = memory * y_t + (1-memory) * x_t, the larger the short term
                memory, the more weight is put on contempory.
            layer_type: fc (fully-connected) or conv (convolutional)
            epsilon:
                denominator min value for preventing division by zero in computing std
            dim: for fc layers, shape is the layer dimension, for conv layers,
                shape is the number of feature maps
        '''

        assert layer_type in ['fc', 'conv']
        self.layer_type = layer_type
        self.epsilon = 1e-6
        self.dim = dim
        self.mem = short_memory

        if self.layer_type == 'fc':
            input_shape = (1, dim)
        elif self.layer_type == 'conv':
            input_shape = (1, 1, 1, dim)

        self.gamma = tf.Variable(tf.random_uniform(input_shape, minval=-0.05, maxval=0.05),
                                                   name=self.__class__.__name__ + '_gamma')
        self.beta = tf.Variable(tf.zeros(input_shape), name=self.__class__.__name__ + '_beta')
        self.moving_mean = 0
        self.moving_var = 1


    def _train_fprop(self, state_below):
        if self.layer_type == 'fc':
            miu = tf.reduce_mean(state_below, 0, keep_dims=True)
            var = tf.reduce_mean((state_below - miu)**2, 0, keep_dims=True)
        elif self.layer_type == 'conv':
            miu = tf.reduce_mean(state_below, (0,1,2), keep_dims=True)
            var = tf.reduce_mean((state_below - miu)**2, (0,2,3), keep_dims=True)
        self.moving_mean = self.mem * miu + (1-self.mem) * self.moving_mean
        self.moving_var = self.mem * var + (1-self.mem) * self.moving_var

        Z = (state_below - self.moving_mean) / tf.sqrt(self.moving_var + self.epsilon)
        return self.gamma * Z + self.beta


    def _test_fprop(self, state_below):
        Z = (state_below - self.moving_mean) / tf.sqrt(self.moving_var + self.epsilon)
        return self.gamma * Z + self.beta

    @property
    def _variables(self):
        return [self.gamma, self.beta]



class TFBatchNormalization(Template):

    def __init__(self, name, epsilon=1e-5, decay=0.9):
        '''
        REFERENCE:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        PARAMS:
            decay: decay for the moving average. Reasonable values for decay are close to 1.0,
                   typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
                   Lower decay value (recommend trying decay=0.9) if model experiences
                   reasonably good training performance but poor validation and/or
                   test performance. Try zero_debias_moving_mean=True for improved stability.
            epislon: small float added to variance to avoid dividing by zero.
        '''
        self.decay = decay
        self.epsilon = epsilon
        self.name = name
        self.first_call = True
        with tf.variable_scope(name):
            try:
                tf.get_variable(name, [])
            except:
                raise Exception('Variable {} exists, choose a different name!'.format(name))

    def _train_fprop(self, state_below):
        with tf.variable_scope(self.name) as scope:
            if not self.first_call:
                scope.reuse_variables()
            out = tf.contrib.layers.batch_norm(state_below,
                              decay=self.decay,
                              updates_collections=None,
                              epsilon=self.epsilon,
                              scale=True,
                              is_training=True,
                              scope='TFBatchNormalization')
        self.first_call = False
        return out


    def _test_fprop(self, state_below):
        with tf.variable_scope(self.name) as scope:
            if not self.first_call:
                scope.reuse_variables()
            out = tf.contrib.layers.batch_norm(state_below,
                              decay=self.decay,
                              updates_collections=None,
                              epsilon=self.epsilon,
                              scale=True,
                              is_training=False,
                              scope='TFBatchNormalization')
        self.first_call = False
        return out


class LRN(Template):

    def __init__(self, depth_radius=None, bias=None, alpha=None, beta=None):
        '''
        DESCRIPTION:
            Local Response Normalization.
            The 4-D input tensor is treated as a 3-D array of 1-D vectors
            (along the last dimension), and each vector is normalized independently.
            Within a given vector, each component is divided by the weighted,
            squared sum of inputs within depth_radius. In detail,

            sqr_sum[a, b, c, d] =   sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
            output = input / (bias + alpha * sqr_sum) ** beta
        PARAMS:
            depth_radius (optional int): defaults to 5. 0-D. Half-width of the 1-D normalization window.
            bias (optional float): Defaults to 1. An offset (usually positive to avoid dividing by 0).
            alpha (optional float): Defaults to 1. A scale factor, usually positive.
            beta (optional float): Defaults to 0.5. An exponent.
        '''
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta


    def _train_fprop(self, state_below):
        return tf.nn.local_response_normalization(state_below, depth_radius=self.depth_radius,
                                                  bias=self.bias, alpha=self.alpha,
                                                  beta=self.beta, name=None)
