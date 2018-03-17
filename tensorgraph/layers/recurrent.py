import tensorflow as tf
from .template import Template
from .merge import Concat
import numpy as np


class DynamicLSTM(Template):

    @Template.init_name_scope
    def __init__(self, num_units, const_seq_len=False, state_is_tuple=True, scope=None):
        '''
        DESCRIPTION:
            DynamicLSTM is for sequences with dynamic length.
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
            const_seq_len (bool): if true, will use a constant sequence
                         length equal to the max_time. state_below will just be
                         singular input instead of tuple.
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.lstm = tf.contrib.rnn.LSTMCell(num_units=num_units, state_is_tuple=state_is_tuple)
        self.const_seq_len = const_seq_len


    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: if state_is_tuple = True (default):
                             return tuple (C, h) each of dimension [batchsize, num_units]
                             C: internal context vector
                             h: last hidden output state
                         if state_is_tuple = False:
                             return tf.concat(1, [C, h]) of dimension [batchsize, 2 * num_units]
        '''
        if self.const_seq_len:
            X_sb, seqlen_sb = state_below, None
        else:
            X_sb, seqlen_sb = state_below

        with tf.variable_scope(self.scope) as scope:
            try:
                bef = set(tf.global_variables())
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=seqlen_sb,
                                                         inputs=X_sb,
                                                         dtype=tf.float32)
                aft = set(tf.global_variables())
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

    @Template.init_name_scope
    def __init__(self, num_units, return_idx=[0,1,2], initial_state=None, state_is_tuple=True, scope=None):
        '''
        DESCRIPTION:
            LSTM is for sequences with fixed length.
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
            initial_state: An initial state for the RNN. If cell.state_size is an integer,
                           this must be a Tensor of appropriate type and shape
                           [batch_size, cell.state_size]. If cell.state_size is a tuple,
                           this should be a tuple of tensors having shapes [batch_size, s]
                           for s in cell.state_size.
            return_idx (list): list of index from the rnn outputs to return from
                           [outputs, context, last_hid], indexes has to fall into
                           [0, 1, 2]
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.lstm = tf.contrib.rnn.LSTMCell(num_units=num_units, state_is_tuple=state_is_tuple)
        self.initial_state = initial_state
        self.state_is_tuple = state_is_tuple
        self.return_idx = return_idx
        assert max(self.return_idx) <= 2 and min(self.return_idx) >= 0, 'indexes \
               does not fall into [outputs, context, last_hid]'
        assert isinstance(self.return_idx, list)


    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: if state_is_tuple = True (default):
                             return tuple (C, h) each of dimension [batchsize, num_units]
                             C: internal context vector
                             h: last hidden output state
                         if state_is_tuple = False:
                             return tf.concat(1, [C, h]) of dimension [batchsize, 2 * num_units]
        '''

        with tf.variable_scope(self.scope) as scope:
            try:
                bef = set(tf.global_variables())
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=None,
                                                         inputs=state_below,
                                                         initial_state = self.initial_state,
                                                         dtype=tf.float32)
                aft = set(tf.global_variables())
                self.var = aft - bef
            except ValueError:
                scope.reuse_variables()
                outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                         sequence_length=None,
                                                         inputs=state_below,
                                                         initial_state = self.initial_state,
                                                         dtype=tf.float32)

        context, last_hid = last_states
        returns = np.asarray([outputs, context, last_hid])
        return returns[self.return_idx]


    @property
    def _variables(self):
        return list(self.var)


class DynamicBiLSTM(Template):

    @Template.init_name_scope
    def __init__(self, fw_num_units, bw_num_units, const_seq_len=False, state_is_tuple=True, scope=None):
        '''
        DESCRIPTION:
            BiDynamicLSTM is for sequences with dynamic length.
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
            const_seq_len (bool): if true, will use a constant sequence
                         length equal to the max_time. state_below will just be
                         singular input instead of tuple.
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.fw_lstm = tf.contrib.rnn.LSTMCell(num_units=fw_num_units, state_is_tuple=state_is_tuple)
            self.bw_lstm = tf.contrib.rnn.LSTMCell(num_units=bw_num_units, state_is_tuple=state_is_tuple)
        self.const_seq_len = const_seq_len


    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
        RETURN:
            outputs (tuple): ([batchsize, max_time, fw_num_units],
                              [batchsize, max_time, bw_num_units])
                              collect the outputs at each time step
            last_states: if state_is_tuple = True (default):
                             return tuple (C, h) each of dimension [batchsize, num_units]
                             C: internal context vector
                             h: last hidden output state
                         if state_is_tuple = False:
                             return tf.concat(1, [C, h]) of dimension [batchsize, 2 * num_units]
        '''
        if self.const_seq_len:
            seqlen_sb = None
            X_sb = state_below
        else:
            X_sb, seqlen_sb = state_below

        with tf.variable_scope(self.scope) as scope:
            try:
                bef = set(tf.global_variables())
                outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_lstm,
                                                                       cell_bw=self.bw_lstm,
                                                                       sequence_length=seqlen_sb,
                                                                       inputs=X_sb,
                                                                       dtype=tf.float32)
                aft = set(tf.global_variables())
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


class Seq2Seq(Template):

    @Template.init_name_scope
    def __init__(self, num_units, state_is_tuple=True, scope=None):
        '''
        DESCRIPTION:
            Seq2Seq model
        PARAMS:
            scope (str): scope for the cells. For RNN with the same scope name,
                         the rnn cell will be reused.
        '''
        if scope is None:
            self.scope = self.__class__.__name__
        else:
            self.scope = scope
        with tf.variable_scope(self.scope):
            self.lstm = tf.contrib.rnn.LSTMCell(num_units=num_units, state_is_tuple=state_is_tuple)


    def _train_fprop(self, state_below):
        '''
        PARAMS:
            state_below: [batchsize, max_time, fea_dim]
            encoder_inputs:
            decoder_inputs:
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: if state_is_tuple = True (default):
                             return tuple (C, h) each of dimension [batchsize, num_units]
                             C: internal context vector
                             h: last hidden output state
                         if state_is_tuple = False:
                             return tf.concat(1, [C, h]) of dimension [batchsize, 2 * num_units]
        '''
        encoder_inputs, decoder_inputs = state_below

        with tf.variable_scope(self.scope) as scope:
            try:
                bef = set(tf.global_variables())
                outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                                                            decoder_inputs,
                                                                            self.cell,
                                                                            dtype=tf.float32)
                aft = set(tf.global_variables())
                self.var = aft - bef
            except ValueError:
                scope.reuse_variables()
                outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                                                            decoder_inputs,
                                                                            self.cell,
                                                                            dtype=tf.float32)
        return outputs, states


    @property
    def _variables(self):
        return list(self.var)



class BiLSTM_Merge(Template):

    @Template.init_name_scope
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

    @Template.init_name_scope
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
