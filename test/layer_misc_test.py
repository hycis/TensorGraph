
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.layers import OneHot
import numpy as np

def test_OneHot():
    X1 = tf.placeholder('int32', [5, 6, 7])
    X2 = tf.placeholder('int32', [5, 6, 7, 8])
    seq = tg.Sequential()
    seq.add(OneHot(onehot_size=3))

    y1 = seq.train_fprop(X1)
    y2 = seq.train_fprop(X2)

    with tf.Session() as sess:
        print(sess.run(y1, feed_dict={X1:np.random.random_integers(0, 2, [5,6,7])}).shape)
        print(sess.run(y2, feed_dict={X2:np.random.random_integers(0, 2, [5,6,7,8])}).shape)

if __name__ == '__main__':
    
    test_OneHot()
