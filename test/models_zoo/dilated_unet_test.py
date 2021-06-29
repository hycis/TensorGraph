import os
import sys
dirt = sys.path[0]
dirt = os.path.abspath(os.path.join(dirt, '../..'))
sys.path.insert(0, dirt)
import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.echocardiac.dilated_unet.model import Dilated_Unet
from tensorgraph.models_zoo.echocardiac.dilated_unet.train import train


def test_model():
    X_train = np.random.rand(5, 128, 128, 1)
    y_train = np.random.rand(5, 128, 128, 1)
    _, h, w, c = X_train.shape

    with tf.Graph().as_default():
        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, h, w, c])
        seq = Dilated_Unet()
        train(seq, X_ph, y_ph, X_train, y_train)

if __name__ == '__main__':
    test_model()
