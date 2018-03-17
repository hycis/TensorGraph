
import tensorflow as tf


# def init_name_scope(self, func):
#     def decorated(obj, *args, **kwargs):
#         with tf.name_scope(self.__class__.__name__) as self.scope:
#             return func(self, *args, **kwargs)
#     return decorated
#
# def train_name_scope(func):
#     def decorated(self, *args, **kwargs):
#         with tf.name_scope(self.scope):
#             with tf.name_scope('train'):
#                 return func(self, *args, **kwargs)
#     return decorated
#
# def test_name_scope(func):
#     def decorated(self, *args, **kwargs):
#         with tf.name_scope(self.scope):
#             with tf.name_scope('test'):
#                 return func(self, *args, **kwargs)
#     return decorated
#

# class init_name_scope(object):
#     def __init__(self, fun):
#         self.func = fun
#     def __call__(self, *args, **kwargs):
#         with tf.name_scope(self.__class__.__name__) as self.scope:
#             self.func(*args, **kwargs)
#
#
# class test_name_scope(object):
#     def __init__(self, fun):
#         self.func = fun
#     def __call__(self, *args, **kwargs):
#         with tf.name_scope(self.scope):
#             with tf.name_scope('test'):
#                 self.func(*args, **kwargs)
#
#
# class train_name_scope(object):
#     def __init__(self, fun):
#         self.func = fun
#     def __call__(self, *args, **kwargs):
#         with tf.name_scope(self.scope):
#             with tf.name_scope('train'):
#                 self.func(*args, **kwargs)


class ScopeDeco(object):

    @classmethod
    def init_name_scope(cls, func):
        def decorated(self, *args, **kwargs):
            with tf.name_scope(self.__class__.__name__) as self.scope:
                return func(self, *args, **kwargs)
        return decorated

    @classmethod
    def train_name_scope(cls, func):
        def decorated(self, *args, **kwargs):
            with tf.name_scope(self.scope):
                with tf.name_scope('train'):
                    return func(self, *args, **kwargs)
        return decorated

    @classmethod
    def test_name_scope(cls, func):
        def decorated(self, *args, **kwargs):
            with tf.name_scope(self.scope):
                with tf.name_scope('test'):
                    return func(self, *args, **kwargs)
        return decorated



class Template(ScopeDeco):

    @ScopeDeco.init_name_scope
    def __init__(self, *args, **kwargs):
        pass

    def _train_fprop(self, state_below):
        raise NotImplementedError()

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)

    @ScopeDeco.train_name_scope
    def train_fprop(self, state_below):
        return self._train_fprop(state_below)

    @ScopeDeco.test_name_scope
    def test_fprop(self, state_below):
        '''Defines the forward propogation through the layer during testing,
           defaults to the same as train forward propogation
        '''
        return self._test_fprop(state_below)

    @property
    def _variables(self):
        '''Defines the trainable parameters in the layer
           RETURN: list of Variables
        '''
        return []
