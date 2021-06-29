
import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.hed_modified.model import HED_Modified
from tensorgraph.models_zoo.hed_modified.train import train


def test_model():
    D, H, W = 16, 64, 64
    X_train = np.random.rand(6, D, H, W, 1)
    y_train = np.random.rand(6, D, H, W, 1)
    _, d, h, w, c = X_train.shape

    with tf.Graph().as_default():
        X_ph = tf.placeholder('float32', [None, d, h, w, c])
        y_ph = tf.placeholder('float32', [None, d, h, w, c])

        seq = HED_Modified(channels=1, side_features=1, output_shape=(D,H,W), output_channels=1, droprate=0.03)
        train(seq, X_ph, y_ph, X_train, y_train)


if __name__ == '__main__':
    test_model()
