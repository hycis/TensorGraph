import tensorflow as tf
from layers.misc import Sum


class StartNode(object):
    def __init__(self, input_vars):
        assert isinstance(input_vars, list)
        self.input_vars = input_vars


class HiddenNode(object):
    def __init__(self, prev, input_merge_mode=Sum(), layers=[]):
        assert isinstance(prev, list)
        assert isinstance(layers, list)
        '''
        PARAMS:
            input_merge_mode(layer): the way to merge the multiple inputs coming into this hidden node
            layers(list): the sequential layers within the node
            prev(list): previous nodes to link to
        '''
        self.input_merge_mode = input_merge_mode
        self.prev = prev
        self.layers = layers
        self.input_vars = []

    def _fprop(self, mode, state):
        assert len(self.input_vars) > 0
        assert mode in ['_train_fprop', '_test_fprop']
        for layer in self.layers:
            state = getattr(layer, mode)(state)
        return [state]

    def train_fprop(self):
        state = self.input_merge_mode._train_fprop(self.input_vars)
        return self._fprop('_train_fprop', state)

    def test_fprop(self):
        state = self.input_merge_mode._test_fprop(self.input_vars)
        return self._fprop('_test_fprop', state)


class EndNode(object):
    def __init__(self, prev, input_merge_mode=Sum()):
        assert isinstance(prev, list)
        self.input_merge_mode = input_merge_mode
        self.prev = prev
        self.input_vars = []

    def train_fprop(self):
        return [self.input_merge_mode._train_fprop(self.input_vars)]

    def test_fprop(self):
        return self.train_fprop()
