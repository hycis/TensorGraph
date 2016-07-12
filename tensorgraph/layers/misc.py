import tensorflow as tf
from template import Template


class Flatten(Template):

    def _train_fprop(self, state_below):
        shape = state_below.get_shape().as_list()
        return tf.reshape(state_below, [shape[0], np.prod(shape[1:])])


class Reshape(object):

    def __init__(self, shape):
        self.shape = shape

    def _train_fprop(self, X):
        return tf.reshape(X, self.shape)


class Embedding(Template):
    def __init__(self, cat_dim=None, encode_dim=None, W=None, zero_pad=False):
        """
        DESCRIPTION:
            embedding
        PARAM:
            cat_dim (int): number of categories
            encode_dim (int): dense encoding of the categories
            W (tensor variable): embedding of 2D tensor matrix
            zero_pad (bool): whether should initialize zero embedding for sequence
                with zero paddings
        """

        self.cat_dim = cat_dim
        self.encode_dim = encode_dim
        self.W = W

        if self.W is None:
            embed = tf.random_normal([self.cat_dim, self.encode_dim], stddev=0.1)
            if zero_pad:
                embed[0] *= 0
            self.W = tf.Variable(embed, name='W')


    def _train_fprop(self, state_below):
        '''
        state_below is a list of indices
        '''
        return tf.gather(self.W, state_below)

    @property
    def _variables(self):
        return [self.W]
