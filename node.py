import tensorflow as tf


class StartNode(object):
    def __init__(self, input_vars):
        assert isinstance(input_vars, list)
        self.input_vars = input_vars


class HiddenNode(object):
    def __init__(self, prev, layers):
        assert isinstance(prev, list)
        assert isinstance(layers, list)
        self.prev = prev
        self.layers = layers
        self.input_vars = []

    def _fprop(self, mode):
        assert len(self.input_vars) > 0
        assert mode in ['_train_fprop', '_test_fprop']
        if len(self.input_vars) == 1:
            out = getattr(self.layers[0], mode)(*self.input_vars)
        else:
            out = getattr(self.layers[0], mode)(self.input_vars)
        for layer in self.layers[1:]:
            out = getattr(layer, mode)(out)
        return [out]

    def train_fprop(self):
        return self._fprop('_train_fprop')

    def test_fprop(self):
        return self._fprop('_test_fprop')


class EndNode(object):
    def __init__(self, prev, outmode='sum'):
        assert isinstance(prev, list)
        assert outmode in ['sum', 'concat']
        self.outmode = outmode
        self.prev = prev
        self.input_vars = []

    def train_fprop(self):
        if self.outmode == 'sum':
            return [tf.add_n(self.input_vars)]
        elif self.outmode == 'concat':
            return [tf.concat(1, self.input_vars)]

    def test_fprop(self):
        return self.train_fprop()
