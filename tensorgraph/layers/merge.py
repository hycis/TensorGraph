
import tensorflow as tf


class Merge(object):
    '''
    Merge layer is used to merge the list of states from layer below into one state
    '''
    def _train_fprop(self, state_list):
        '''state_list (list): list of states to be merged
        '''
        raise NotImplementedError()

    def _test_fprop(self, state_list):
        '''Defines the forward propogation through the layer during testing,
           defaults to the same as train forward propogation
        '''
        return self._train_fprop(state_list)

    @property
    def _variables(self):
        '''Defines the trainable parameters in the layer
        '''
        return []


class Concat(Merge):
    def __init__(self, axis=1):
        self.axis = axis

    def _train_fprop(self, state_list):
        return tf.concat(self.axis, state_list)


class Mean(Merge):
    def _train_fprop(self, state_list):
        return tf.add_n(state_list) / len(state_list)


class Sum(Merge):
    def _train_fprop(self, state_list):
        return tf.add_n(state_list)


class NoChange(Merge):
    def _train_fprop(self, state_list):
        return state_list


class Multiply(Merge):
    def _train_fprop(self, state_list):
        out = state_list[0]
        for state in state_list[1:]:
            out = tf.mul(out, state)
        return out


class Select(Merge):
    def __init__(self, index=0):
        self.index = index

    def _train_fprop(self, state_list):
        return state_list[self.index]


class SequenceMask(Merge):
    def __init__(self, maxlen):
        '''
        DESCRIPTION:
            Mask the sequence of shape [batchsize, max_seq_len, :, ..] at the
            second dimension by using a mask tensor representing the first N
            positions of each row.
            Example:
                mask = tf.sequence_mask(lengths=[1, 3, 2], maxlen=5) =
                       [[True, False, False, False, False],
                       [True, True, True, False, False],
                       [True, True, False, False, False]]
                y = X * mask
        '''
        self.maxlen = maxlen

    def _train_fprop(self, state_list):
        assert len(state_list) == 2
        state_below, seqlen = state_list
        mask = tf.to_float(tf.sequence_mask(seqlen, self.maxlen))
        num_dim = len(state_below.get_shape())
        for _ in range(num_dim-2):
            mask = tf.expand_dims(mask, -1)
        return state_below * mask


class MaskSoftmax(Merge):
    def _train_fprop(self, state_list):
        '''The softmax is apply to units that is not masked
           state_list : [state_below, seqlen]
                state_below (2d tf tensor): shape = [batchsize, layer_dim]
                seqlen (1d tf tensor): shape = [batchsize]
                example:
                    state_below = 3 x 5 matrix
                    seqlen = [2, 1, 4]
        '''
        assert len(state_list) == 2
        state_below, seqlen = state_list
        assert len(seqlen.get_shape()) == 1
        shape = state_below.get_shape()
        assert len(shape) == 2, 'state below dimenion {} != 2'.format(len(shape))
        mask = tf.to_float(tf.sequence_mask(seqlen, shape[-1]))
        exp = tf.exp(state_below) * mask
        exp_sum = tf.reduce_sum(exp, axis=1)
        return tf.div(exp, tf.expand_dims(exp_sum, -1))
