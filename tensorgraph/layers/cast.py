import tensorflow as tf
from .template import BaseLayer


class ToFloat(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.to_float(state_below, name='ToFloat')


class ToInt32(BaseLayer):
    def _train_fprop(self, state_below):
        return tf.to_int32(state_below, name='ToInt32')
