import tensorflow as tf

class Sum(object):
    def train_fprop(self, state_list):
        return tf.add_n(state_list)

    def test_fprop(self, state_list):
        return self.train_fprop(state_list)


class NoChangeState(object):

    @staticmethod
    def check_y(y):
        '''Check if the output list contains one element or a list, if contains
           only one element, return the element, if contains more than one element,
           returns the entire list.
        '''
        if len(y) == 1:
            return y[0]
        elif len(y) > 1:
            return y
        else:
            raise Exception('{} is empty or not a list'.format(y))

    def train_fprop(self, state_list):
        return NoChangeState.check_y(state_list)

    def test_fprop(self, state_list):
        return self.train_fprop(state_list)


class StartNode(object):
    def __init__(self, input_vars):
        '''
        StartNode defines the input to the graph

        Args:
            input_vars (list of tensors): the input tensors to the graph, which
                can be a placeholder or output of another graph or a tensor.
        '''
        assert isinstance(input_vars, list)
        self.input_vars = input_vars


class HiddenNode(object):
    def __init__(self, prev, input_merge_mode=Sum(), layers=[]):
        '''
        HiddenNode encapsulates a list of layers, it can be connected to a StartNode
        or another HiddenNode

        Args:
            input_merge_mode(tensorgraph.layers.Merge): ``Merge`` Layer for merging
                the multiple inputs coming into this hidden node
            layers(list): the sequential layers within the node
            prev(list): list of previous nodes to link to
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
            layer.__init_var__(state)
            state = layer.train_fprop(state)
        return [state]


    def test_fprop(self):
        if len(self.input_vars) == 0:
            return []
        state = self.input_merge_mode.test_fprop(self.input_vars)
        for layer in self.layers:
            layer.__init_var__(state)
            state = layer.test_fprop(state)
        return [state]


    @property
    def variables(self):
        var = []
        for layer in self.layers:
            var += layer._variables
        return var


class EndNode(object):
    def __init__(self, prev, input_merge_mode=NoChangeState()):
        '''
        EndNode is where we want to get the output from the graph. It can be
        connected to a HiddenNode or a StartNode.

        Args:
            input_merge_mode(tensorgraph.layers.Merge): ``Merge`` Layer for merging
                the multiple inputs coming into this hidden node
            prev(list): list of previous nodes to link to
        '''
        assert isinstance(prev, list)
        self.input_merge_mode = input_merge_mode
        self.prev = prev
        self.input_vars = []


    def train_fprop(self):
        return [self.input_merge_mode.train_fprop(self.input_vars)]

    def test_fprop(self):
        return [self.input_merge_mode.test_fprop(self.input_vars)]
