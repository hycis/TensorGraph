import tensorflow as tf
from template import Template


class Flatten(Template):

    def _train_fprop(self, state_below):
        shape = state_below.get_shape().as_list()
        return tf.reshape(state_below, [shape[0], np.prod(shape[1:])])
