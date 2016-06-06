

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


    @staticmethod
    def _output(node, mode):
        if node.__class__.__name__ == 'StartNode':
            return node.input_vars
        input_vars = []
        for pnode in node.prev:
            input_vars += Graph._output(pnode, mode)
        node.input_vars = input_vars
        return getattr(node, mode)()


    def train_fprop(self):
        outs = []
        for node in self.end:
            outs += Graph._output(node, 'train_fprop')
        return outs


    def test_fprop(self):
        outs = []
        for node in self.end:
            outs += Graph._output(node, 'test_fprop')
        return outs
