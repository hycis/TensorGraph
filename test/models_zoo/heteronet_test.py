import sys
import numpy as np
import tensorflow as tf
from tensorgraph.models_zoo.heteronet.model import HeteroNet
from tensorgraph.models_zoo.heteronet.train import HeteroTrain

def test_heteromodel():
    tf.reset_default_graph()
    model  = HeteroNet()
    X_ph   = tf.placeholder(tf.float32, [1, 4, 320,320,1])
    y_ph   = tf.placeholder(tf.float32, [1, 30])
    
    Model = HeteroTrain(model, X_ph, X_ph, X_ph, y_ph)
    print('training')
    for i in range(15):
        X_train = np.random.rand(1, 4, 320,320,1)
        y_train = np.random.randint(0,2,[1,30])
        Model.train(feed_dict={X_ph:X_train, y_ph:y_train})

if __name__ == '__main__':
    test_heteromodel()
