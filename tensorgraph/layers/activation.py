import tensorflow as tf
from template import Template


class RELU(Template):
    def _train_fprop(self, state_below):
        return tf.nn.relu(state_below)


class RELU6(Template):
    def _train_fprop(self, state_below):
        return tf.nn.relu6(state_below)


class ELU(Template):
    def _train_fprop(self, state_below):
        return tf.nn.elu(state_below)


class Softplus(Template):
    def _train_fprop(self, state_below):
        return tf.nn.softplus(state_below)


class Softsign(Template):
    def _train_fprop(self, state_below):
        return tf.nn.softsign(state_below)


class Tanh(Template):
    def _train_fprop(self, state_below):
        return tf.nn.tanh(state_below)


class Sigmoid(Template):
    def _train_fprop(self, state_below):
        return tf.nn.sigmoid(state_below)


class Tanh(Template):
    def _train_fprop(self, state_below):
        return tf.tanh(state_below)


class Softmax(Template):
    def _train_fprop(self, state_below):
        return tf.nn.softmax(state_below)
