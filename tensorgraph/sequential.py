

class Sequential(object):

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def pop(self, index):
        return self.layers.pop(index)

    def train_fprop(self, input_state, layers=None):
        if layers is None:
            layers = xrange(len(self.layers))
        for i in layers:
            layer_output = self.layers[i]._train_fprop(input_state)
            input_state = layer_output
        return input_state

    def test_fprop(self, input_state, layers=None):
        if layers is None:
            layers = xrange(len(self.layers))
        for i in layers:
            layer_output = self.layers[i]._test_fprop(input_state)
            input_state = layer_output
        return input_state
