

def output(node, mode):
    if node.__class__.__name__ == 'StartNode':
        return node.input_vars
    input_vars = []
    for pnode in node.prev:
        input_vars += output(pnode, mode)
    node.input_vars = input_vars
    return getattr(node, mode)()


class Graph(object):

    def __init__(self, start, end):
        assert isinstance(start, list)
        assert isinstance(end, list)
        self.start = start
        self.end = end


    def train_fprop(self):
        outs = []
        for node in self.end:
            outs += output(node, 'train_fprop')
        return outs


    def test_fprop(self):
        outs = []
        for node in self.end:
            outs += output(node, 'test_fprop')
        return outs
