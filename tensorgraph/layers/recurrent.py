import tensorflow as tf
from .template import Template


class DynamicLSTM(Template):

    def __init__(self, num_units):
        '''
        DESCRIPTION:
            DynamicLSTM is for sequences with dynamic length.
        '''
        self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units, state_is_tuple=True)
        

    def _train_fprop(self, state_below):
        '''
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: [batchsize, num_units] the output of the last lstm iteration
        '''
        X_sb, len_sb = state_below
        outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                 sequence_length=len_sb,
                                                 inputs=X_sb)
        return outputs, last_states


class LSTM(Template):

    def __init__(self, num_units):
        '''
        DESCRIPTION:
            LSTM is for sequences with fixed length.
        '''
        self.lstm = tf.nn.rnn_cell.LSTMCell(num_units=num_units, state_is_tuple=True)


    def _train_fprop(self, state_below):
        '''
        RETURN:
            outputs: [batchsize, max_time, num_units] collect the outputs at each time step
            last_states: [batchsize, num_units] the output of the last lstm iteration
        '''
        outputs, last_states = tf.nn.dynamic_rnn(cell=self.lstm,
                                                 sequence_length=None,
                                                 inputs=state_below)
        return outputs, last_states


class BiDynamicLSTM(Template):

    def __init__(self, fw_num_units, bw_num_units):
        '''
        DESCRIPTION:
            BiDynamicLSTM is for sequences with dynamic length.
        '''
        self.fw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=fw_num_units, state_is_tuple=True)
        self.bw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=bw_num_units, state_is_tuple=True)


    def _train_fprop(self, state_below):
        '''
        RETURN:
            outputs (tuple): ([batchsize, max_time, fw_num_units],
                              [batchsize, max_time, bw_num_units])
                              collect the outputs at each time step
            last_states (tuple): ([batchsize, fw_num_units], [batchsize, bw_num_units])
                                  the output of the last lstm iteration
        '''
        X_sb, len_sb = state_below
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_lstm,
                                                               cell_bw=self.bw_lstm,
                                                               sequence_length=len_sb,
                                                               inputs=X_sb)
        return outputs, last_states


class BiLSTM(Template):

    def __init__(self, fw_num_units, bw_num_units):
        '''
        DESCRIPTION:
            BiLSTM is for sequences with fixed length.
        '''
        self.fw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=fw_num_units, state_is_tuple=True)
        self.bw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=bw_num_units, state_is_tuple=True)


    def _train_fprop(self, state_below):
        '''
        RETURN:
            outputs (tuple): ([batchsize, max_time, fw_num_units],
                              [batchsize, max_time, bw_num_units])
                              collect the outputs at each time step
            last_states (tuple): ([batchsize, fw_num_units], [batchsize, bw_num_units])
                                  the output of the last lstm iteration
        '''
        outputs, last_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_lstm,
                                                               cell_bw=self.bw_lstm,
                                                               sequence_length=None,
                                                               inputs=state_below)
        return outputs, last_states
