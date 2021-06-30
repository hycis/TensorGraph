
import tensorflow as tf
import tensorgraph as tg
from tensorgraph.layers import Depthwise_Conv2D, Atrous_Conv2D, Conv2D, Conv3D
import numpy as np

def test_Depthwise_Conv2D():

    seq = tg.Sequential()
    seq.add(Depthwise_Conv2D(num_filters=2, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))

    X_ph = tf.placeholder('float32', [None, 100, 100, 5])

    y_sb = seq.train_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_sb, feed_dict={X_ph:np.random.rand(32,100,100,5)})
        print(out.shape)


def test_Conv2D():

    seq = tg.Sequential()
    seq.add(Conv2D(num_filters=2, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))

    X_ph = tf.placeholder('float32', [None, 100, 100, 5])
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_train_sb, feed_dict={X_ph:np.random.rand(32,100,100,5)})
        print(out.shape)


def test_Atrous_Conv2D():

    seq = tg.Sequential()
    seq.add(Atrous_Conv2D(num_filters=2, kernel_size=(3, 3), rate=3, padding='SAME'))

    h, w, c = 100, 300, 5
    X_ph = tf.placeholder('float32', [None, h, w, c])

    y_sb = seq.train_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_sb, feed_dict={X_ph:np.random.rand(32, h, w, c)})
        print(out.shape)
        assert out.shape[1] == h and out.shape[2] == w
    seq = tg.Sequential()
    r = 2
    k = 5
    seq.add(Atrous_Conv2D(num_filters=2, kernel_size=(k, k), rate=r, padding='VALID'))

    h, w, c = 100, 300, 5
    X_ph = tf.placeholder('float32', [None, h, w, c])

    y_sb = seq.train_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_sb, feed_dict={X_ph:np.random.rand(32, h, w, c)})
        print(out.shape)
        assert out.shape[1] == h - 2*int((k+(k-1)*(r-1))/2), out.shape[2] == w - 2*int((w+(w-1)*(r-1))/2)


def test_Conv3D():
    seq = tg.Sequential()
    seq.add(Conv3D(num_filters=2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME'))
    X_ph = tf.placeholder('float32', [None, 10, 10, 10, 5])
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_train_sb, feed_dict={X_ph:np.random.rand(32,10,10,10,5)})
        print(out.shape)


if __name__ == '__main__':
    test_Conv2D()
    test_Depthwise_Conv2D()
    test_Atrous_Conv2D()
    test_Conv3D()
