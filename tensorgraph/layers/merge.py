
import tensorflow as tf




class Merge(object):
    '''
    Merge layers is used to merge the list of states from layer below into one state
    '''
    def _train_fprop(self, state_list):
        raise NotImplementedError()

    def _test_fprop(self, state_list):
        '''Defines the forward propogation through the layer during testing,
           defaults to the same as train forward propogation
        '''
        return self._train_fprop(state_list)

    def _variables(self):
        '''Defines the trainable parameters in the layer
        '''
        return []


class Concat(Merge):
    def _train_fprop(self, state_list):
        return tf.concat(1, state_list)


class Mean(Merge):
    def _train_fprop(self, state_list):
        return tf.add_n(state_list) / len(state_list)


class Sum(Merge):
    def _train_fprop(self, state_list):
        return tf.add_n(state_list)
