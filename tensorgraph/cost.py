
import tensorflow as tf

epsilon = 1e-6

def mse(ytrue, ypred):
    return tf.reduce_mean((ytrue - ypred)**2)

def entropy(ytrue, ypred):
    ypred = tf.clip_by_value(ypred, epsilon, 1.0 - epsilon)
    L = -(ytrue * tf.log(ypred) + (1-ytrue) * tf.log(1-ypred))
    return tf.reduce_mean(L)

def binary_f1(ytrue, ypred):
    '''ytrue and ypred is one-hot, and have to be of type int and shape [batchsize, 2]
    since it is binary, the values must be 0 and 1'''
    r = binary_recall(ytrue, ypred)
    p = binary_precision(ytrue, ypred)
    return 2 * p * r / (p + r)

def binary_recall(ytrue, ypred):
    '''ytrue and ypred is one-hot, and have to be of type int and shape [batchsize, 2]
    since it is binary, the values must be 0 and 1'''
    P = tf.reduce_sum(ytrue[:, 1])
    TP = tf.reduce_sum(ypred[:,1] * ytrue[:,1]) # both ypred and ytrue are positives
    return tf.to_float(TP) / tf.to_float(P)

def binary_precision(ytrue, ypred):
    '''ytrue and ypred is one-hot, and have to be of type int and shape [batchsize, 2]
    since it is binary, the values must be 0 and 1'''
    TPnFP = tf.reduce_sum(ypred[:, 1])
    TP = tf.reduce_sum(ypred[:,1] * ytrue[:,1]) # both ypred and ytrue are positives
    return tf.to_float(TP) / tf.to_float(TPnFP)

def hingeloss(ytrue, ypred):
    ypred = tf.clip_by_value(ypred, 0., 1.0)
    L = tf.maximum(0, 1 - ytrue * ypred)
    return tf.reduce_mean(tf.to_float(L))

def error(ytrue, ypred):
    '''ytrue and ypred is 2d'''
    L = tf.not_equal(tf.argmax(ypred, 1), tf.argmax(ytrue, 1))
    return tf.reduce_mean(tf.to_float(L))

def accuracy(ytrue, ypred):
    '''ytrue and ypred is 2d'''
    L = tf.equal(tf.argmax(ypred, 1), tf.argmax(ytrue, 1))
    return tf.reduce_mean(tf.to_float(L))
