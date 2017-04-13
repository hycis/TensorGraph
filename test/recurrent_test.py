import numpy as np
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.layers import LSTM, DynamicLSTM, DynamicBiLSTM


def test_lstm(layer, seq_len, fea_dim):
    x_ph = tf.placeholder('float32', [None, seq_len, fea_dim])

    seq = tg.Sequential()
    seq.add(layer)
    print(seq.total_num_parameters)
    train_sb = seq.train_fprop(x_ph)
    test_sb = seq.test_fprop(x_ph)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {x_ph:np.random.rand(100, seq_len, fea_dim)}
        outputs, last_states = sess.run(train_sb, feed_dict=feed_dict)
        C, h = last_states
        print('outputs: {}'.format(outputs.shape))
        print('last_states: {}, {}'.format(C.shape, h.shape))
        # import pdb; pdb.set_trace()
        sess.run(test_sb, feed_dict=feed_dict)
        print(layer.__class__.__name__ + ' test done!')


def test_dynamic_lstm(layer, seq_len, fea_dim):
    x_ph = tf.placeholder('float32', [None, seq_len, fea_dim])
    seq_ph = tf.placeholder('int64', [None])

    seq = tg.Sequential()
    seq.add(layer)

    train_sb = seq.train_fprop([x_ph, seq_ph])
    test_sb = seq.test_fprop([x_ph, seq_ph])

    print('... total number of parameters', seq.total_num_parameters())

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {x_ph:np.random.rand(100, seq_len, fea_dim),
                     seq_ph:np.random.randint(1, 10, size=100)}
        sess.run(train_sb, feed_dict=feed_dict)
        sess.run(test_sb, feed_dict=feed_dict)
        sess.run(train_sb, feed_dict=feed_dict)
        print(layer.__class__.__name__ + ' test done!')


def test_lstm_cell():
    batch_size = 32
    num_steps = 3
    lstm_size = 2
    words = tf.placeholder(tf.int32, [batch_size, num_steps])
    # import pdb; pdb.set_trace()
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
    # Initial state of the LSTM memory.
    initial_context = context = tf.zeros([batch_size, lstm_size])
    # import pdb; pdb.set_trace()
    hidden = words[:, 0]
    for i in range(num_steps):
        # The value of state is updated after processing each batch of words.
        context, hidden = lstm(words[:, i], state=(context, hidden))

        # The rest of the code.
        # ...

    final_state = state
    import pdb; pdb.set_trace()
    print()


if __name__ == '__main__':
    # test_lstm_cell()
    lstm = LSTM(num_units=21, state_is_tuple=True, scope='lstm')
    # dylstm = DynamicLSTM(num_units=10, scope='dylstm')
    # dybilstm = DynamicBiLSTM(fw_num_units=10, bw_num_units=20, scope='bilstm')
    #
    for layer in [lstm]:
        test_lstm(layer, seq_len=15, fea_dim=8)
    #
    # for layer in [dylstm, dybilstm]:
    #     test_dynamic_lstm(layer, seq_len=15, fea_dim=8)
