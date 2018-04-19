import numpy as np


class Graph(object):
    TRAIN_FPROP = 'train_fprop'
    TEST_FPROP = 'test_fprop'

    def __init__(self, start, end):
        '''
        PARAMS:
            start(list): list of start nodes
            end(list): list of end nodes

        '''
        assert isinstance(start, list)
        assert isinstance(end, list)
        self.start = start
        self.end = end
        for node in self.start:
            assert node.__class__.__name__ == 'StartNode'
        for node in self.end:
            assert node.__class__.__name__ == 'EndNode'
        # nodes visited during train or test fprop
        self.visited_train = {}
        self.visited_test = {}


    def _output(self, node, mode):
        assert node.__class__.__name__ in ['StartNode', 'HiddenNode', 'EndNode']
        if node.__class__.__name__ == 'StartNode':
            if node in self.start:
                return node.input_vars
            else:
                return []
        input_vars = []
        for pnode in node.prev:
            if mode == Graph.TRAIN_FPROP:
                # check if the train mode of hidden node has been visited
                if pnode not in self.visited_train:
                    output = self._output(pnode, mode)
                    input_vars += output
                    self.visited_train[pnode] = output
                else:
                    input_vars += self.visited_train[pnode]

            elif mode == Graph.TEST_FPROP:
                # check if the test mode of hidden node has been visited
                if pnode not in self.visited_test:
                    output = self._output(pnode, mode)
                    input_vars += output
                    self.visited_test[pnode] = output
                else:
                    input_vars += self.visited_test[pnode]
            else:
                raise Exception('unknown mode: {}'.format(mode))

        node.input_vars = input_vars
        return getattr(node, mode)()


    def train_fprop(self):
        outs = []
        for node in self.end:
            outs += self._output(node, Graph.TRAIN_FPROP)
        return outs


    def test_fprop(self):
        outs = []
        for node in self.end:
            outs += self._output(node, Graph.TEST_FPROP)
        return outs

    @property
    def variables(self):
        def outvars(node):
            var = []
            if node.__class__.__name__ == 'StartNode':
                return var
            for pnode in node.prev:
                var += outvars(pnode)
            if node.__class__.__name__ == 'HiddenNode':
                var += node.variables
            return var

        var = []
        for node in self.end:
            var += outvars(node)
        return list(set(var))


    def total_num_parameters(self):
        count = 0
        for var in self.variables:
            count += int(np.prod(var.get_shape()))
        return count
