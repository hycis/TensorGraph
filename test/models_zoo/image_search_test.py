import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.image_search.model import Image_Search_Model
from tensorgraph.models_zoo.image_search.train import train


def test_model():
    X_train = np.random.rand(8, 24, 160, 160, 5)          #the total number is 8, should be very big.   5 is the channels number(for t1,t2,dwi,etc)
    y_train = np.random.randint(low=0, high=5, size=8)    #the labels: 0,1,2,3,4 （means totally 5 kinds of diseases）
    _, d, h, w, c = X_train.shape
    with tf.Graph().as_default():
        X_ph = tf.placeholder('float32', [None, d, h, w, c])
        y_ph = tf.placeholder('float32', [None])
        seq = Image_Search_Model()
        train(seq, X_ph, y_ph, X_train, y_train)


if __name__ == '__main__':
    test_model()


