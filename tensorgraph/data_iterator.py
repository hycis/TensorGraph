import numpy as np


class DataIterator(object):
    def __init__(self, *data, **params):
        '''
        PARAMS:
            fullbatch (bool): decides if the number of examples return after every
                              iteration should be always a full batch.
        '''
        self.data = data
        self.batchsize = params['batchsize']
        if 'fullbatch' in params:
            self.fullbatch = params['fullbatch']
        else:
            self.fullbatch = False

    def __iter__(self):
        self.first = 0
        return self

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, key):
        outs = []
        for val in self.data:
            outs.append(val[key])
        return self.__class__(*outs, batchsize=self.batchsize, fullbatch=self.fullbatch)


class SequentialIterator(DataIterator):
    '''
    batchsize = 3
    [0, 1, 2], [3, 4, 5], [6, 7, 8]
    '''
    def next(self):
        if self.fullbatch and self.first+self.batchsize > len(self):
            raise StopIteration()
        elif self.first >= len(self):
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
        if self.fullbatch and self.first+self.batchsize > len(self):
            raise StopIteration()
        elif self.first >= len(self):
            raise StopIteration()

        outs = []
        for val in self.data:
            outs.append(val[self.first:self.first+self.batchsize])
        self.first += self.step
        return outs
