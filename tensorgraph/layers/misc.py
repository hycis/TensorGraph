import tensorflow as tf
from template import Template


class Flatten(Template):

    def _train_fprop(self, state_below):
        shape = state_below.get_shape().as_list()
        return tf.reshape(state_below, [shape[0], np.prod(shape[1:])])


class Embedding(Template)
    def __init__(self, cat_dim=None, encode_dim=None, W=None, include_zero=False):
        """
        DESCRIPTION:
            embedding
        PARAM:
            cat_dim (int): number of categories
            encode_dim (int): dense encoding of the categories
            W (tensor variable): embedding of 2D tensor matrix
            include_zero (bool): to include a null embedding row, if true, there
                will be an additional embedding row (cat_dim+1), index 0 will be
                for uncategorized features, real embedding starts from index 1.
        """

        self.cat_dim = cat_dim
        self.encode_dim = encode_dim
        self.W = W

        if include_zero:
            if self.W is None:
                embed = tf.random_normal([self.cat_dim+1, self.encode_dim], stddev=0.1)
                embed[0] = 0
                self.W = tf.Variable(embed, name='W')
        else:
            if self.W is None:
                self.W = tf.Variable(tf.random_normal([self.cat_dim, self.encode_dim], stddev=0.1), name='W')


    def _train_fprop(self, state_below):
        '''
        state_below is a list of indices
        '''
        return tf.gather(self.W, state_below)

    @property
    def _variables(self):
        return [self.W]
