
import tensorflow as tf

epsilon = 1e-6

def mse(ytrue, ypred):
    return tf.reduce_mean((ytrue - ypred)**2)

def entropy(ytrue, ypred):
    ypred = tf.clip_by_value(ypred, epsilon, 1.0 - epsilon)
    L = -(ytrue * tf.log(ypred) + (1-ytrue) * tf.log(1-ypred))
    return tf.reduce_mean(L)
