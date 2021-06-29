"""
Purpose: Training image search model
"""
import numpy as np
import tensorflow as tf
import os
from ...trainobject import train as mytrain
from ...cost import entropy, accuracy
#import triplet_or_hist_loss as th_loss
from . import triplet_or_hist_loss as th_loss


def train(seq, X_ph, y_ph, X_train, y_train):
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)

    #Set the target for triplet loss
    target_list = list([1,2])  #suppose that the first two disease is the target.
    target = tf.constant(target_list, dtype = tf.int32)
    target_size = len(target_list)

    loss_choose = 'triplet'
    if loss_choose == 'triplet':
        train_cost_sb, fraction_positive_triplets = th_loss.triplet_loss(y_ph, y_train_sb, alpha=0.3,
                                                                              target=target_list, labels_size=1,
                                                                              target_size=target_size, penalize_ratio=0.2)
    elif loss_choose == 'histogram':
        train_cost_sb = th_loss.histogram_loss(y_ph, y_train_sb, target=target_list, labels_size=1,
                                                        target_size=target_size, penalize_ratio=0.2)

    optimizer = tf.train.AdamOptimizer(0.0001)
    test_accu_sb = accuracy(y_ph, y_test_sb)
    with tf.Session() as sess:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        writer = tf.summary.FileWriter(this_dir + '/tensorboard', sess.graph)
        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=train_cost_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=1,
                percent_decrease=0, train_valid_ratio=[1,1],
                batchsize=4, randomize_split=False)
        writer.close()
