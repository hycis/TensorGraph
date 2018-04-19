import tensorflow as tf


class Sum(object):
    def train_fprop(self, state_list):
        return tf.add_n(state_list)

    def test_fprop(self, state_list):
        return self.train_fprop(state_list)

class StartNode(object):
    def __init__(self, input_vars):
        assert isinstance(input_vars, list)
        self.input_vars = input_vars


class HiddenNode(object):
    def __init__(self, prev, input_merge_mode=Sum(), layers=[]):
        '''
        PARAMS:
            input_merge_mode(layer): the way to merge the multiple inputs coming into this hidden node
            layers(list): the sequential layers within the node
            prev(list): previous nodes to link to
        '''
        assert isinstance(prev, list)
        assert isinstance(layers, list)
        self.input_merge_mode = input_merge_mode
        self.prev = prev
        self.layers = layers
        self.input_vars = []

    def train_fprop(self):
        if len(self.input_vars) == 0:
            return []
        state = self.input_merge_mode.train_fprop(self.input_vars)
        for layer in self.layers:
            state = layer.train_fprop(state)
        return [state]

    def test_fprop(self):
        if len(self.input_vars) == 0:
            return []
        state = self.input_merge_mode.test_fprop(self.input_vars)
        for layer in self.layers:
            state = layer.test_fprop(state)
        return [state]

    @property
    def variables(self):
        var = []
        for layer in self.layers:
            var += layer._variables
        return var


class EndNode(object):
    def __init__(self, prev, input_merge_mode=Sum()):
        '''
        DESCRIPTION:
            EndNode is where we want to get the output from the graph.
        PARAMS:
            input_merge_mode(layer): the way to merge the multiple inputs coming into this hidden node
            prev(list): previous nodes to link to
        '''
        assert isinstance(prev, list)
        self.input_merge_mode = input_merge_mode
        self.prev = prev
        self.input_vars = []

    def train_fprop(self):
        return [self.input_merge_mode.train_fprop(self.input_vars)]

    def test_fprop(self):
        return [self.input_merge_mode.test_fprop(self.input_vars)]
