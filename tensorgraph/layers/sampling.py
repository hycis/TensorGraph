import tensorflow as tf
from .template import BaseLayer

class OneSample(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, dim):
        '''
        Description:
            multinomial sample one output from the softmax probability

        Args:
            dim (int): layer dimension
        '''
        self.diag = tf.diag(tf.ones(dim))


    def _train_fprop(self, state_below):
        samples = tf.multinomial(state_below, num_samples=1)
        samples = tf.squeeze(samples)
        return tf.gather(self.diag, samples)
