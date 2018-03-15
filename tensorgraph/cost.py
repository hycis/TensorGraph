
import tensorflow as tf
import sys

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

def image_f1(ytrue, ypred):
    r = image_recall(ytrue, ypred)
    p = image_precision(ytrue, ypred)
    pnr = tf.clip_by_value(p + r, epsilon, float('inf'))
    f1 = 2 * p * r / pnr
    return tf.reduce_mean(f1)

def image_recall(ytrue, ypred):
    ndims = len(ytrue.get_shape())
    assert ndims > 1
    P = tf.reduce_sum(ytrue, axis=list(range(1, ndims)))
    P = tf.clip_by_value(tf.to_float(P), epsilon, float('inf'))
    TP = tf.reduce_sum(ytrue * ypred, axis=list(range(1, ndims)))
    return tf.to_float(TP) / tf.to_float(P)

def image_precision(ytrue, ypred):
    ndims = len(ytrue.get_shape())
    assert ndims > 1
    TPnFP = tf.reduce_sum(ypred, axis=list(range(1, ndims)))
    TPnFP = tf.clip_by_value(tf.to_float(TPnFP), epsilon, float('inf'))
    TP = tf.reduce_sum(ypred * ytrue, axis=list(range(1, ndims))) # both ypred and ytrue are positives
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

def smooth_iou(ytrue, ypred):
    ytrue = tf.reshape(ytrue, [-1, tf.reduce_prod(tf.shape(ytrue)[1:])])
    ypred = tf.reshape(ypred, [-1, tf.reduce_prod(tf.shape(ypred)[1:])])
    I = tf.reduce_sum(ytrue * ypred, axis=1)
    y_area = tf.reduce_sum(ytrue, axis=1)
    ypred_area = tf.reduce_sum(ypred, axis=1)
    IOU = I * 1.0 / (y_area + ypred_area - I + 1e-6)
    return tf.reduce_mean(IOU)

def iou(ytrue, ypred, threshold=0.5):
    ytrue = tf.reshape(ytrue, [-1, tf.reduce_prod(tf.shape(ytrue)[1:])])
    ypred = tf.reshape(ypred, [-1, tf.reduce_prod(tf.shape(ypred)[1:])])
    ypred = tf.to_float(ypred > threshold)
    I = tf.reduce_sum(ytrue * ypred, axis=1)
    y_area = tf.reduce_sum(ytrue, axis=1)
    ypred_area = tf.reduce_sum(ypred, axis=1)
    IOU = I * 1.0 / (y_area + ypred_area - I + 1e-6)
    return tf.reduce_mean(IOU)
