
from tensorgraph.layers import Linear, LinearMasked, SparseLinear
import tensorgraph as tg
import tensorflow as tf
import numpy as np

def test_linear():
    seq = tg.Sequential()
    seq.add(Linear(this_dim=100))
    seq.add(LinearMasked(this_dim=200, mask=np.zeros(200)))
    seq.add(Linear(this_dim=10))


    X_ph = tf.placeholder('float32', [None, 100])

    y_sb = seq.train_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_sb, feed_dict={X_ph:np.random.rand(32,100)})
        print(out.shape)

def test_SparseLinear():
    seq = tg.Sequential()
    seq.add(SparseLinear(prev_dim=10, this_dim=300, batchsize=8))
    seq.add(Linear(this_dim=10))

    idx_ph = tf.placeholder('int32', [None, None])
    val_ph = tf.placeholder('float32', [None])
    y_sb = seq.train_fprop([idx_ph, val_ph])
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_sb, feed_dict={idx_ph:[[0, 0], [1, 2]], val_ph:[5,6]})
        print(out.shape)

if __name__ == '__main__':
    test_linear()
    test_SparseLinear()
