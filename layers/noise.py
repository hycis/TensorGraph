
import tensorflow as tf
from template import Template


class Dropout(Template):
    def _train_fprop(self, state_below):
        return tf.nn.dropout(state_below)
