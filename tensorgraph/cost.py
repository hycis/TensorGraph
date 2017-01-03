
import tensorflow as tf

epsilon = 1e-6



def mse(ytrue, ypred):
    return tf.reduce_mean((ytrue - ypred)**2)

def entropy(ytrue, ypred):
    ypred = tf.clip_by_value(ypred, epsilon, 1.0 - epsilon)
    L = -(ytrue * tf.log(ypred) + (1-ytrue) * tf.log(1-ypred))
    return tf.reduce_mean(L)

def binary_f1(ytrue, ypred):
    r = binary_recall(ytrue, ypred)
    p = binary_precision(ytrue, ypred)
    return 2 * p * r / (p + r)

def binary_recall(ytrue, ypred):
    P = tf.reduce_sum(ytrue[:, 1])
    TP = tf.reduce_sum(ypred[:,1] * ytrue[:,1]) # both ypred and ytrue are positives
    return tf.to_float(TP) / tf.to_float(P)

def binary_precision(ytrue, ypred):
    TPnFP = tf.reduce_sum(ypred[:, 1])
    TP = tf.reduce_sum(ypred[:,1] * ytrue[:,1]) # both ypred and ytrue are positives
    return tf.to_float(TP) / tf.to_float(TPnFP)

def hingeloss(ytrue, ypred):
    ypred = tf.clip_by_value(ypred, 0., 1.0)
    L = tf.maximum(0, 1 - ytrue * ypred)
    return tf.reduce_mean(L)

def error(ytrue, ypred):
    L = tf.not_equal(tf.argmax(ypred, 1), tf.argmax(ytrue, 1))
    return tf.reduce_mean(L)

def accuracy(ytrue, ypred):
    L = tf.equal(tf.argmax(ypred, 1), tf.argmax(ytrue, 1))
    return tf.reduce_mean(L)
