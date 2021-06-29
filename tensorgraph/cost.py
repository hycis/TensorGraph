
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
    '''
    ytrue and ypred is one-hot, and have to be of type int and shape [batchsize, 2]
    since it is binary, the values must be 0 and 1
    '''
    r = binary_recall(ytrue, ypred)
    p = binary_precision(ytrue, ypred)
    return 2 * p * r / (p + r)

def binary_recall(ytrue, ypred):
    '''
    ytrue and ypred is one-hot, and have to be of type int and shape [batchsize, 2]
    since it is binary, the values must be 0 and 1
    '''
    P = tf.reduce_sum(ytrue[:, 1])
    TP = tf.reduce_sum(ypred[:,1] * ytrue[:,1]) # both ypred and ytrue are positives
    return tf.to_float(TP) / tf.to_float(P)

def binary_precision(ytrue, ypred):
    '''
    ytrue and ypred is one-hot, and have to be of type int and shape [batchsize, 2]
    since it is binary, the values must be 0 and 1
    '''
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

def generalised_dice(y_true, y_pred, smooth=1.0):
    '''
    Dice Loss Cost Function, where smaller groundtruth channels is penalised more heavily
    y_true: tensor of shape (?, D, H, W, c)
    '''
    pos_count = tf.reduce_sum(y_true, axis=(1,2,3))     + tf.constant(0.1, dtype='float32')
    w         = 1.0 / tf.multiply(pos_count, pos_count)
    
    rp = tf.reduce_sum((y_true)*(y_pred), axis=(1,2,3))  # Intersection of prediction and ground-truth
    r  = tf.reduce_sum(y_true,            axis=(1,2,3))  # ground-truth
    p  = tf.reduce_sum(y_pred,            axis=(1,2,3))  # prediction

    wrp = tf.multiply(w, rp)                             # Weighted intersection
    wr  = tf.multiply(w, r)                              # Weighted ground-truth
    wp  = tf.multiply(w, p)                              # Weighted prediction
    
    neg_count = tf.reduce_sum((1.0 - y_true), axis=(1,2,3))     + tf.constant(0.1, dtype='float32')
    w_inv     = 1.0 / tf.multiply(neg_count, neg_count)
    
    rp_inv = tf.reduce_sum((1.0 - y_true)*(1.0 - y_pred), axis=(1,2,3))
    r_inv  = tf.reduce_sum((1.0 - y_true),                axis=(1,2,3))
    p_inv  = tf.reduce_sum((1.0 - y_pred),                axis=(1,2,3))

    wrp_inv = tf.multiply(w_inv, rp_inv)
    wr_inv  = tf.multiply(w_inv, r_inv)
    wp_inv  = tf.multiply(w_inv, p_inv)
    
    wrp_concat = tf.concat([wrp, wrp_inv], -1)
    wr_concat  = tf.concat([wr,  wr_inv],  -1)
    wp_concat  = tf.concat([wp,  wp_inv],  -1)
    
    dice_loss = 1.0 - (2 * tf.reduce_sum(wrp_concat, axis=1) / (tf.reduce_sum(wr_concat, axis=1) + tf.reduce_sum(wp_concat, axis=1)))
    
    return tf.reduce_mean(dice_loss)
    
def mean_dice(y_true, y_pred, smooth=1.0):
    '''
    Dice Loss Cost Function, where all output channels are penalised equally
    y_true: tensor of shape (?, D, H, W, c)
    '''
    rp = tf.reduce_sum((y_true)*(y_pred), axis=(1,2,3))  # Intersection of prediction and ground-truth
    r  = tf.reduce_sum(y_true,            axis=(1,2,3))  # ground-truth
    p  = tf.reduce_sum(y_pred,            axis=(1,2,3))  # prediction

    dice_loss = 1.0 - (2 * tf.reduce_sum(rp, axis=0) / (tf.reduce_sum(r, axis=0) + tf.reduce_sum(p, axis=0)))
    
    return tf.reduce_mean(dice_loss)

def inv_dice(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    ones_array = tf.ones_like(y_true)
    yI_pred = ones_array - y_pred
    yI_true = ones_array - y_true
    intersection = tf.reduce_sum(yI_pred * yI_true)
    coefficient = (2.0*intersection +smooth) / (tf.reduce_sum(yI_pred)+tf.reduce_sum(yI_true) +smooth)
    loss = 1.0 - coefficient
    return(loss)
