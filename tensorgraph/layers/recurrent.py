import tensorflow as tf
from .template import Template
from .merge import Concat


class DynamicLSTM(Template):

    def __init__(self, num_units, scope=None):
        '''
        DESCRIPTION:
            DynamicLSTM is for sequences with dynamic length.
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units, state_is_tuple=False)



    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: [batchsize, num_units] the output of the last lstm iteration
        '''
        X_sb, seqlen_sb = state_below
        with tf.variable_scope(self.scope) as scope:

            try:
                bef = set(tf.all_variables())
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=seqlen_sb,
                                                         inputs=X_sb,
                                                         dtype=tf.float32)
                aft = set(tf.all_variables())
                self.var = aft - bef
            except ValueError:
                scope.reuse_variables()
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=seqlen_sb,
                                                         inputs=X_sb,
                                                         dtype=tf.float32)
        return outputs, last_states

    @property
    def _variables(self):
        return list(self.var)



class LSTM(Template):

    def __init__(self, num_units, scope=None):
        '''
        DESCRIPTION:
            LSTM is for sequences with fixed length.
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units, state_is_tuple=False)


    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: [batchsize, num_units] the output of the last lstm iteration
        '''

        with tf.variable_scope(self.scope) as scope:
            try:
                bef = set(tf.all_variables())
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=None,
                                                         inputs=state_below,
                                                         dtype=tf.float32)
                aft = set(tf.all_variables())
                self.var = aft - bef
            except ValueError:
                scope.reuse_variables()
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=None,
                                                         inputs=state_below,
                                                         dtype=tf.float32)

        return outputs, last_states


    @property
    def _variables(self):
        return list(self.var)


class DynamicBiLSTM(Template):

    def __init__(self, fw_num_units, bw_num_units, scope=None):
        '''
        DESCRIPTION:
            BiDynamicLSTM is for sequences with dynamic length.
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.fw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=fw_num_units, state_is_tuple=False)
            self.bw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=bw_num_units, state_is_tuple=False)


    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
        RETURN:
            outputs (tuple): ([batchsize, max_time, fw_num_units],
                              [batchsize, max_time, bw_num_units])
                              collect the outputs at each time step
            last_states (tuple): ([batchsize, fw_num_units], [batchsize, bw_num_units])
                                  the output of the last lstm iteration
        '''
        X_sb, seqlen_sb = state_below

        with tf.variable_scope(self.scope) as scope:
            try:
                bef = set(tf.all_variables())
                outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_lstm,
                                                                       cell_bw=self.bw_lstm,
                                                                       sequence_length=seqlen_sb,
                                                                       inputs=X_sb,
                                                                       dtype=tf.float32)
                aft = set(tf.all_variables())
                self.var = aft - bef
            except ValueError:
                scope.reuse_variables()
                outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_lstm,
                                                                       cell_bw=self.bw_lstm,
                                                                       sequence_length=seqlen_sb,
                                                                       inputs=X_sb,
                                                                       dtype=tf.float32)

        return outputs, last_states

    @property
    def _variables(self):
        return list(self.var)


class BiLSTM_Merge(Template):

    def __init__(self, merge_mode=Concat(axis=2)):
        '''
        DESCRIPTION:
            merge the outputs from BiLSTM and DynamicLSTM, default to concat
        '''
        self.merge_mode = merge_mode


    def _train_fprop(self, state_below):
        outputs, last_states = state_below
        return self.merge_mode._train_fprop(outputs)


    def _test_fprop(self, state_below):
        outputs, last_states = state_below
        return self.merge_mode._test_fprop(outputs)


class BiLSTM_Last_Merge(Template):

    def __init__(self, merge_mode=Concat(axis=1)):
        '''
        DESCRIPTION:
            select last state from BiLSTM and DynamicLSTM and performs merge,
            default to concat
        '''
        self.merge_mode = merge_mode


    def _train_fprop(self, state_below):
        outputs, last_states = state_below
        return self.merge_mode._train_fprop(last_states)


    def _test_fprop(self, state_below):
        outputs, last_states = state_below
        return self.merge_mode._test_fprop(last_states)
