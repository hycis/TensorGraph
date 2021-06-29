
import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.densenet.model import MyDenseNet
from tensorgraph.models_zoo.densenet.train import train


def test_model():
    X_train = np.random.rand(5, 32, 32, 1)
    y_train = np.random.rand(5, 1)
    _, h, w, c = X_train.shape
    _, nclass = y_train.shape

    with tf.Graph().as_default():
        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, nclass])

        seq = MyDenseNet(nclass)
        train(seq, X_ph, y_ph, X_train, y_train)


if __name__ == '__main__':
    test_model()
