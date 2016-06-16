

class Template(object):

    def _train_fprop(self, state_below):
        raise NotImplementedError()

    def _test_fprop(self, state_below):
        '''Defines the forward propogation through the layer during testing,
           defaults to the same as train forward propogation
        '''
        return self._train_fprop(state_below)

    @property
    def _variables(self):
        '''Defines the trainable parameters in the layer
           RETURN: list of Variables
        '''
        return []
