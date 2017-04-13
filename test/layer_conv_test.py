
import tensorflow as tf
import tensorgraph as tg
from tensorgraph.layers import Depthwise_Conv2D
import numpy as np

def test_Depthwise_Conv2d():

    seq = tg.Sequential()
    seq.add(Depthwise_Conv2D(input_channels=5, num_filters=2, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))

    X_ph = tf.placeholder('float32', [None, 100, 100, 5])

    y_sb = seq.train_fprop(X_ph)
    with tf.Session() as  sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        out = sess.run(y_sb, feed_dict={X_ph:np.random.rand(32,100,100,5)})
        print(out.shape)


if __name__ == '__main__':
    test_Depthwise_Conv2d()
