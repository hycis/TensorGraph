from template import Template


class Iterative(Template):

    def __init__(self, sequential, num_iter):
        self.sequential = sequential
        self.num_iter = num_iter


    def _train_fprop(self, state_below):
        for i in range(self.num_iter):
            out = self.sequential.train_fprop(state_below)
            state_below = out + state_below
        return out


    def _test_fprop(self, state_below):
        for i in range(self.num_iter):
            out = self.sequential.test_fprop(state_below)
            state_below = out + state_below
        return out
