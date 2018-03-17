import numpy as np

class Sequential(object):

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def pop(self, index):
        return self.layers.pop(index)

    def train_fprop(self, input_state, layers=None):
        if layers is None:
            layers = range(len(self.layers))
        for i in layers:
            layer_output = self.layers[i].train_fprop(input_state)
            input_state = layer_output
        return input_state

    def test_fprop(self, input_state, layers=None):
        if layers is None:
            layers = range(len(self.layers))
        for i in layers:
            layer_output = self.layers[i].test_fprop(input_state)
            input_state = layer_output
        return input_state

    @property
    def variables(self):
        var = []
        for layer in self.layers:
            var += layer._variables
        return list(set(var))

    def total_num_parameters(self):
        count = 0
        for var in self.variables:
            count += int(np.prod(var.get_shape()))
        return count
