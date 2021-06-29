import tensorflow as tf
from .template import BaseLayer


class RELU(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.relu(state_below)


class RELU6(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.relu6(state_below)


class LeakyRELU(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, leak=0.2):
        self.leak = leak

    def _train_fprop(self, state_below):
        return tf.maximum(state_below, self.leak*state_below)


class ELU(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.elu(state_below)


class Softplus(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.softplus(state_below)


class Softsign(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.softsign(state_below)


class Tanh(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.tanh(state_below)


class Sigmoid(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.sigmoid(state_below)


class Tanh(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.tanh(state_below)


class Softmax(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.nn.softmax(state_below)
