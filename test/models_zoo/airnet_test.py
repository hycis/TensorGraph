
import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.airnet.model import AIRNet
from tensorgraph.models_zoo.airnet.train import train


def test_model():
    X1_train = np.random.rand(5, 8, 160, 160, 1)
    X2_train = np.random.rand(5, 8, 160, 160, 1)
    y_train = np.random.rand(5, 12)
    _, d, h, w, c = X1_train.shape
    _, n = y_train.shape

    with tf.Graph().as_default():
        X1_ph = tf.placeholder('float32', [None, d, h, w, c])
        X2_ph = tf.placeholder('float32', [None, d, h, w, c])
        y_ph = tf.placeholder('float32', [None, n])

        seq = AIRNet()
        train(seq, X1_ph, X2_ph, y_ph, X1_train, X2_train, y_train)


if __name__ == '__main__':
    test_model()
