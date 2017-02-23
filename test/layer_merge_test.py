
import tensorflow as tf
import tensorgraph as tg
from tensorgraph.layers import SequenceMask, MaskSoftmax
import numpy as np

def test_SequenceMask():
    X_ph = tf.placeholder('float32', [None, 5, 6, 7])
    seq_ph = tf.placeholder('int32', [None])

    X_sn = tg.StartNode(input_vars=[X_ph])
    seq_sn = tg.StartNode(input_vars=[seq_ph])

    merge_hn = tg.HiddenNode(prev=[X_sn, seq_sn], input_merge_mode=SequenceMask(maxlen=5))

    out_en = tg.EndNode(prev=[merge_hn])

    graph = tg.Graph(start=[X_sn, seq_sn], end=[out_en])

    y_train_sb = graph.train_fprop()
    y_test_sb = graph.test_fprop()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = {X_ph:np.random.rand(3,5,6,7), seq_ph:[2,3,4]}
        y_train = sess.run(y_train_sb, feed_dict=feed_dict)[0]
        y_test = sess.run(y_test_sb, feed_dict=feed_dict)[0]
        assert y_train.sum() == y_test.sum()
        assert y_train[0, :2].sum() > 0 and y_train[0, 2:].sum() == 0
        assert y_train[1, :3].sum() > 0 and y_train[1, 3:].sum() == 0
        assert y_train[2, :4].sum() > 0 and y_train[2, 4:].sum() == 0
        print('test passed!')


def test_MaskSoftmax():
    X_ph = tf.placeholder('float32', [None, 20])
    seq_ph = tf.placeholder('int32', [None])

    X_sn = tg.StartNode(input_vars=[X_ph])
    seq_sn = tg.StartNode(input_vars=[seq_ph])

    merge_hn = tg.HiddenNode(prev=[X_sn, seq_sn], input_merge_mode=MaskSoftmax())

    y_en = tg.EndNode(prev=[merge_hn])

    graph = tg.Graph(start=[X_sn, seq_sn], end=[y_en])
    y_sb, = graph.train_fprop()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {X_ph:np.random.rand(3, 20),
                     seq_ph:[5, 8, 0]}
        out = sess.run(y_sb, feed_dict=feed_dict)
        assert out[0][5:].sum() == 0
        assert out[0][:5].sum() == 1
        assert out[1][8:].sum() == 0
        assert out[1][:8].sum() == 1
        assert out[2].sum() == 0
        print('test passed!')


if __name__ == '__main__':
    # test_SequenceMask()
    test_MaskSoftmax()
