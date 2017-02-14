import tensorflow as tf
from .template import Template


class ToFloat(Template):
    def _train_fprop(self, state_below):
        return tf.to_float(state_below, name='ToFloat')


class ToInt32(Template):
    def _train_fprop(self, state_below):
        return tf.to_int32(state_below, name='ToInt32')
