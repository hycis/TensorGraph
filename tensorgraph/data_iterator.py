import numpy as np


class DataIterator(object):
    def __init__(self, *data, **params):
        self.data = data
        self.batchsize = params['batchsize']

    def __iter__(self):
        self.first = 0
        return self

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, key):
        outs = []
        for val in self.data:
            outs.append(val[key])
        return outs


class SequentialIterator(DataIterator):
    '''
    batchsize = 3
    [0, 1, 2], [3, 4, 5], [6, 7, 8]
    '''
    def next(self):
        if self.first >= len(self):
            raise StopIteration()
        outs = []
        for val in self.data:
            outs.append(val[self.first:self.first+self.batchsize])
        self.first += self.batchsize
        return outs


class StepIterator(DataIterator):
    '''
    batchsize = 3
    step = 1
    [0, 1, 2], [1, 2, 3], [2, 3, 4]
    '''
    def __init__(self, *data, **params):
        super(self, StepIterator).__init__(self, *data, **params)
        self.step = params['step']

    def next(self):
        if self.first >= len(self):
            raise StopIteration()
        outs = []
        for val in self.data:
            outs.append(val[self.first:self.first+self.batchsize])
        self.first += self.step
        return outs
