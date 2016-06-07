
import tensorflow as tf
from template import Template


class Dropout(Template):

    def __init__(self, dropout_below=0.5):
        '''
        PARAMS:
            dropout_below(float): probability of the inputs from the layer below
            been masked out
        '''
        self.dropout_below = dropout_below


    def _test_fprop(self, state_below):
        """
        DESCRIPTION:
            Since input is already scaled up during training, therefore during
            testing, we don't need to scale the inputs again
        """
        return state_below


    def _train_fprop(self, state_below):
        """
        DESCRIPTION:
            Applies dropout to the layer during training with probability keep_prob,
            outputs the input element scaled up by 1 / keep_prob
        PARAMS:
            keep_prob: probability of keeping the neuron active
        """
        return tf.nn.dropout(state_below, keep_prob=1-self.dropout_below)
