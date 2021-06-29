
import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.attention_unet.model import Attention_UNet
from tensorgraph.models_zoo.attention_unet.train import train


def test_model():
    X_train = np.random.rand(5, 128, 128, 1)
    y_train = np.random.rand(5, 128, 128, 1)
    _, h, w, c = X_train.shape

    with tf.Graph().as_default():
        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, h, w, c])

        seq = Attention_UNet(input_shape=(h, w, c))
        train(seq, X_ph, y_ph, X_train, y_train)

if __name__ == '__main__':
    test_model()
