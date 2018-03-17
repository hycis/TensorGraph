
import tensorflow as tf
from .template import Template


class Dropout(Template):

    @Template.init_name_scope
    def __init__(self, dropout_below=0.5, noise_shape=None):
        '''
        PARAMS:
            dropout_below(float): probability of the inputs from the layer below
                been masked out
            noise_shape (list or tuple): shape of the noise: example [-1, 2, -1] which applies
                noise to the second dimension only
        '''
        self.dropout_below = dropout_below
        self.noise_shape = noise_shape

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
        if self.noise_shape is not None:
            assert len(state_below.get_shape()) == len(self.noise_shape)
            noise_shape = []
            for i, v in enumerate(self.noise_shape):
                if v == -1 or v is None:
                    noise_shape.append(tf.shape(state_below)[i])
                else:
                    noise_shape.append(v)
            self.noise_shape = noise_shape

        return tf.nn.dropout(state_below, keep_prob=1-self.dropout_below,
                             noise_shape=self.noise_shape)
