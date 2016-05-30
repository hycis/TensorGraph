import tensorflow as tf
import numpy as np
from template import Template

class Flatten(Template):

    def _train_fprop(self, state_below):
        shape = state_below.get_shape().as_list()
        return tf.reshape(state_below, [shape[0], np.prod(shape[1:])])


class Concat(Template):
    def _train_fprop(self, state_below):
        return tf.concat(1, state_below)


class Mean(Template):
    def _train_fprop(self, state_below):
        return tf.add_n(state_below) / len(state_below)


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
            layer_type: fc or conv
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

        self.gamma = tf.Variable(tf.random_uniform(input_shape, minval=-0.05, maxval=0.05), name='gamma')
        self.beta = tf.Variable(tf.zeros(input_shape), name='beta')
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


    def _variables(self):
        return self.gamma, self.beta
