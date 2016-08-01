

class Graph(object):

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


    def _output(self, node, mode):
        assert node.__class__.__name__ in ['StartNode', 'HiddenNode', 'EndNode']
        if node.__class__.__name__ == 'StartNode':
            if node in self.start:
                return node.input_vars
            else:
                return []
        input_vars = []
        for pnode in node.prev:
            input_vars += self._output(pnode, mode)
        node.input_vars = input_vars
        return getattr(node, mode)()


    def train_fprop(self):
        outs = []
        for node in self.end:
            outs += self._output(node, 'train_fprop')
        return outs


    def test_fprop(self):
        outs = []
        for node in self.end:
            outs += self._output(node, 'test_fprop')
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
