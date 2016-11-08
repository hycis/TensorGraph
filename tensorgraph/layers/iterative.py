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
