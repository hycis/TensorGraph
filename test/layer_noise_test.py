
import tensorflow as tf
<<<<<<< HEAD
import tensorgraph as tg
from tensorgraph.layers import Linear, Dropout
=======
import tensorgraphx as tg
from tensorgraphx.layers import Linear, Dropout
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
import numpy as np

def test_Dropout():
    X_ph = tf.placeholder('float32', [None, 32])

    seq = tg.Sequential()
<<<<<<< HEAD
    seq.add(Linear(20))
=======
    seq.add(Linear(32, 20))
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
    seq.add(Dropout(0.2, noise_shape=[-1, 20]))



    out = seq.train_fprop(X_ph)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(out, feed_dict={X_ph:np.random.rand(1, 32)})
        print(out)
        print(out.shape)

def test_dropout():
    X_ph = tf.placeholder('float', [None, 5, 10])
    out = tf.nn.dropout(X_ph, keep_prob=0.5, noise_shape=[tf.shape(X_ph)[0], 5, 1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(out, feed_dict={X_ph:np.random.rand(3, 5, 10)})
        print(out)
        print(out.shape)


if __name__ == '__main__':
    test_Dropout()
    # test_dropout()
