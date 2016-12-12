from .template import Template
from .merge import Sum
import tensorgraph as tg

class Iterative(Template):

    def __init__(self, sequential, num_iter, input_merge_mode=Sum()):
        assert(isinstance(sequential, tg.Sequential))
        self.sequential = sequential
        self.num_iter = num_iter
        self.input_merge_mode = input_merge_mode


    def _train_fprop(self, state_below):
        for i in range(self.num_iter):
            out = self.sequential.train_fprop(state_below)
            state_below = self.input_merge_mode._train_fprop([out, state_below])
        return out


    def _test_fprop(self, state_below):
        for i in range(self.num_iter):
            out = self.sequential.test_fprop(state_below)
            state_below = self.input_merge_mode._test_fprop([out, state_below])
        return out


class ResNet(Template):

    def __init__(self, num_blocks):
        '''
        num_blocks (int): number of resnet blocks
        '''
        self.blocks = []
        for _ in range(num_blocks):
            layers = [] # put the layers inside
            self.blocks.append(layers)


    def _train_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._train_fprop(out)
            state_below = out + state_below
        return state_below


    def _test_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._test_fprop(out)
            state_below = out + state_below
        return state_below
