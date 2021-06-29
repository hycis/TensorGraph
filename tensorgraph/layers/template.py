
import tensorflow as tf
from ..graph import Graph
from ..node import StartNode, HiddenNode, EndNode
from functools import wraps

class ScopeDeco(object):

    @classmethod
    def init_name_scope(cls, func):
        @wraps(func)
        def decorated(self, *args, **kwargs):
            if not hasattr(self, 'scope'):
                with tf.name_scope(self.__class__.__name__) as self.scope:
                    return func(self, *args, **kwargs)
            elif not hasattr(self, '__func_visited_by_fprop__'):
                self.__func_visited_by_fprop__ = True
                with tf.name_scope(self.scope):
                    return func(self, *args, **kwargs)
        return decorated


    @classmethod
    def fprop_name_scope(cls, func):
        @wraps(func)
        def decorated(self, *args, **kwargs):
            if hasattr(self, 'scope'):
                with tf.name_scope(self.scope + func.__name__):
                    return func(self, *args, **kwargs)
            else:
                print('{}: scope not initiated for {}'.format(func.__name__, self.__class__.__name__))
                return func(self, *args, **kwargs)
        return decorated



class Template(ScopeDeco):

    @ScopeDeco.init_name_scope
    def __init__(self, *args, **kwargs):
        pass

    @ScopeDeco.init_name_scope
    def __init_var__(self, state_below):
        '''Define variables which requires input information from state_below,
           this is called during forward propagation
        '''
        pass

    def _train_fprop(self, state_below):
        raise NotImplementedError()

    def _test_fprop(self, state_below):
        '''Defines the forward propogation through the layer during testing,
           defaults to the same as train forward propogation
        '''
        return self._train_fprop(state_below)

    @ScopeDeco.fprop_name_scope
    def train_fprop(self, state_below):
        return self._train_fprop(state_below)

    @ScopeDeco.fprop_name_scope
    def test_fprop(self, state_below):
        return self._test_fprop(state_below)

    @property
    def _variables(self):
        '''Defines the trainable parameters in the layer
           Returns: list of Variables
        '''
        return []


class BaseLayer(Template):
    '''renaming of Template to BaseLayer'''
    pass


class BaseModel(Template):

    @staticmethod
    def check_y(y):
        '''Check if the output list contains one element or a list, if contains
           only one element, return the element, if contains more than one element,
           returns the entire list.
        '''
        if len(y) == 1:
            return y[0]
        elif len(y) > 1:
            return y
        else:
            raise Exception('{} is empty or not a list'.format(y))


    def _train_fprop(self, *state_belows):
        self.startnode.input_vars = state_belows
        graph = Graph(start=[self.startnode], end=[self.endnode])
        y = graph.train_fprop()
        return BaseModel.check_y(y)


    def _test_fprop(self, *state_belows):
        self.startnode.input_vars = state_belows
        graph = Graph(start=[self.startnode], end=[self.endnode])
        y = graph.test_fprop()
        return BaseModel.check_y(y)


    def train_fprop(self, *state_below):
        return self._train_fprop(*state_below)

    def test_fprop(self, *state_below):
        return self._test_fprop(*state_below)
