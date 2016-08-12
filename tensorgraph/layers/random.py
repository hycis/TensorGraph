from template import Template
import tensorflow as tf
import numpy as np

class OneSample(Template):

    def __init__(self, dim):
        '''
        DESCRIPTION:
            multinomial sample one output from the softmax probability
        PARAM:
            dim (int): layer dimension
        '''
        self.diag = tf.diag(np.ones(dim))


    def _train_fprop(self, state_below):
        samples = tf.multinomial(state_below, num_samples=1)
        samples = tf.squeeze(samples)
        return tf.cast(tf.gather(self.diag, samples), state_below.dtype)
