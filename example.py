


'''
start1 = DataNode(X=(T.matrix()))
start2 = DataNode(X=(T.matrix()))

h1 = HiddenNode(prev=[start1, start2], model=Convolution2D())
h2 = HiddenNode(prev=[start2], model=Linear())
h3 = HiddenNode(prev=[h1, h2], model=Linear())
e1 = EndNode(prev=[h3], y=(T.matrix()))
e2 = EndNode(prev=[h3], y=(T.matrix()))

graph = Graph(start=[start1, start2], end=[e1, e2])
data = MultiInputsData(X=[x1, x2], y=[y1, y2])

DistributedFlow(model=graph, data=data, gpus=[1,2,3])

'''

from mozi.model import Model

class Graph(Model):

    def __init__(self, start, end):
        assert isinstance(start, tuple)
        assert isinstance(end, tuple)
        self.models = []
        self.input_var = []
        self.output_var
        self.layers = []
        for node in end:
            mod = fprop_layers()



    def test_fprop(self, input_state):

        rstate = []
        for state, mod in zip(, self.models):
            rstate.append(mod.test_fprop(state))


        return input_state, test_layers_stats


    def train_fprop(self, input_state, layers=None):
        train_layers_stats = []
        if layers is None:
            layers = xrange(len(self.layers))
        for i in layers:
            layer_output = self.layers[i]._train_fprop(input_state)
            stats = self.layers[i]._layer_stats(input_state, layer_output)
            input_state = layer_output
            class_name = self.layers[i].__class__.__name__
            stats = [(str(i)+'_'+class_name+'_'+a, b) for (a,b) in stats]
            train_layers_stats += stats

        return input_state, train_layers_stats




    graph = Graph(start=[start1, start2], end=[e1, e2])
