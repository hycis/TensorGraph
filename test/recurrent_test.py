import numpy as np
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.layers import LSTM, DynamicLSTM, DynamicBiLSTM


def test_lstm(layer, seq_len, fea_dim):
    x_ph = tf.placeholder('float32', [None, seq_len, fea_dim])

    seq = tg.Sequential()
    seq.add(layer)

    train_sb = seq.train_fprop(x_ph)
    test_sb = seq.test_fprop(x_ph)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {x_ph:np.random.rand(100, seq_len, fea_dim)}
        sess.run(train_sb, feed_dict=feed_dict)
        sess.run(test_sb, feed_dict=feed_dict)
        sess.run(train_sb, feed_dict=feed_dict)
        print(layer.__class__.__name__ + ' test done!')


def test_dynamic_lstm(layer, seq_len, fea_dim):
    x_ph = tf.placeholder('float32', [None, seq_len, fea_dim])
    seq_ph = tf.placeholder('int64', [None])

    seq = tg.Sequential()
    seq.add(layer)

    train_sb = seq.train_fprop([x_ph, seq_ph])
    test_sb = seq.test_fprop([x_ph, seq_ph])
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        feed_dict = {x_ph:np.random.rand(100, seq_len, fea_dim),
                     seq_ph:np.random.randint(1, 10, size=100)}
        sess.run(train_sb, feed_dict=feed_dict)
        sess.run(test_sb, feed_dict=feed_dict)
        sess.run(train_sb, feed_dict=feed_dict)
        print(layer.__class__.__name__ + ' test done!')


if __name__ == '__main__':
    lstm = LSTM(num_units=10)
    dylstm = DynamicLSTM(num_units=10)
    dybilstm = DynamicBiLSTM(fw_num_units=10, bw_num_units=20)

    for layer in [lstm]:
        test_lstm(layer, seq_len=15, fea_dim=8)

    for layer in [dylstm, dybilstm]:
        test_dynamic_lstm(layer, seq_len=15, fea_dim=8)
