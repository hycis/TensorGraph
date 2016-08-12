from template import Template
import tensorflow as tf

class OneSample(Template):

    def __init__(self, dim):
        '''
        DESCRIPTION:
            multinomial sample one output from the softmax probability
        PARAM:
            dim (int): layer dimension
        '''
        self.diag = tf.diag(tf.ones(dim))


    def _train_fprop(self, state_below):
        samples = tf.multinomial(state_below, num_samples=1)
        samples = tf.squeeze(samples)
        return tf.gather(self.diag, samples)
