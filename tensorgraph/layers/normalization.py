import tensorflow as tf
from tensorflow.python.layers.normalization import BatchNormalization as TFBatchNorm
from .template import Template


class L2_Normalize(Template):

    def __init__(self, dim):
        '''dim (int or list of ints): dimension to normalize'''
        self.dim = dim

    def _train_fprop(self, state_below):
        return tf.nn.l2_normalize(state_below, dim=self.dim)


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


class BatchNormalization(Template):

    def __init__(self, input_shape):
        '''
        REFERENCE:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        PARAMS:
            input_shape (list): shape of the input, do not need the batch dimension

        # To use this normalization, apply update ops below to update the mean and variance
        from tensorflow.python.framework import ops
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
        with ops.control_dependencies(update_ops):
            train_op = optimizer.minimize(train_cost_sb)
        '''
        self.bn = TFBatchNorm()
        self.bn.build(input_shape=[None] + list(input_shape))


    def _train_fprop(self, state_below):
        return self.bn.apply(state_below, training=True)


    def _test_fprop(self, state_below):
        return self.bn.apply(state_below, training=False)


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
