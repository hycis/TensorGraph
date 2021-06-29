import numpy as np
import tensorflow as tf
import os
from ....trainobject import train as mytrain
from ....cost import entropy, accuracy


def train(seq, X_ph, y_ph, X_train, y_train):
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)
    train_cost_sb = entropy(y_ph, y_train_sb)
    optimizer = tf.train.AdamOptimizer(0.0001)
    test_accu_sb = accuracy(y_ph, y_test_sb)
    with tf.Session() as sess:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        writer = tf.summary.FileWriter(this_dir + '/tensorboard', sess.graph)
        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=1,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=1, randomize_split=False)
        writer.close()