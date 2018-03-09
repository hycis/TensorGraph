# -*- coding: utf-8 -*-

"""
From the paper: STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
https://arxiv.org/pdf/1412.6806.pdf
"""

from __future__ import division, print_function, absolute_import

from tensorgraph.layers import Conv2D, RELU, MaxPooling, LRN, Tanh, Dropout, \
                               Softmax, Flatten, Linear, AvgPooling, \
                               Lambda, BatchNormalization, IdentityBlock, \
                               TransitionLayer, DenseNet
from tensorgraph.utils import same, valid, same_nd, valid_nd
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, mse
from tensorgraph.dataset import Mnist, Cifar10
from tensorflow.python.framework import ops
import numpy as np


def model(nclass, h, w, c):
    with tf.name_scope('Cifar10AllCNN'):
        seq = tg.Sequential()
        seq.add(Conv2D(input_channels=c, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(BatchNormalization(input_shape=[h,w,96]))

        seq.add(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
        seq.add(BatchNormalization(input_shape=[h,w,96]))

        seq.add(Conv2D(input_channels=96, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(BatchNormalization(input_shape=[h,w,192]))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(BatchNormalization(input_shape=[h,w,192]))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=nclass, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
        seq.add(BatchNormalization(input_shape=[h,w,nclass]))

        seq.add(AvgPooling(poolsize=(h, w), stride=(1,1), padding='VALID'))
        seq.add(Flatten())
        seq.add(Softmax())
    return seq


def train():
    learning_rate = 0.001
    batchsize = 64
    max_epoch = 300
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=None,
                         percent_decrease=0)

    X_train, y_train, X_test, y_test = Cifar10(contrast_normalize=False, whiten=False)
    _, h, w, c = X_train.shape
    _, nclass = y_train.shape

    seq = model(nclass=nclass, h=h, w=w, c=c)
    iter_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
    iter_test = tg.SequentialIterator(X_test, y_test, batchsize=batchsize)

    X_ph = tf.placeholder('float32', [None, h, w, c])
    y_ph = tf.placeholder('float32', [None, nclass])

    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)

    train_cost_sb = entropy(y_ph, y_train_sb)
    test_cost_sb = entropy(y_ph, y_test_sb)
    test_accu_sb = accuracy(y_ph, y_test_sb)

    # required for BatchNormalization layer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    with ops.control_dependencies(update_ops):
        train_ops = optimizer.minimize(train_cost_sb)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        best_valid_accu = 0
        for epoch in range(max_epoch):
            print('epoch:', epoch)
            pbar = tg.ProgressBar(len(iter_train))
            ttl_train_cost = 0
            ttl_examples = 0
            print('..training')
            for X_batch, y_batch in iter_train:
                feed_dict = {X_ph:X_batch, y_ph:y_batch}
                _, train_cost = sess.run([train_ops,train_cost_sb] , feed_dict=feed_dict)
                ttl_train_cost += len(X_batch) * train_cost
                ttl_examples += len(X_batch)
                pbar.update(ttl_examples)
            mean_train_cost = ttl_train_cost/float(ttl_examples)
            print('\ntrain cost', mean_train_cost)

            ttl_valid_cost = 0
            ttl_valid_accu = 0
            ttl_examples = 0
            pbar = tg.ProgressBar(len(iter_test))
            print('..validating')
            for X_batch, y_batch in iter_test:
                feed_dict = {X_ph:X_batch, y_ph:y_batch}
                valid_cost, valid_accu = sess.run([test_cost_sb, test_accu_sb] , feed_dict=feed_dict)
                ttl_valid_cost += len(X_batch) * valid_cost
                ttl_valid_accu += len(X_batch) * valid_accu
                ttl_examples += len(X_batch)
                pbar.update(ttl_examples)
            mean_valid_cost = ttl_valid_cost/float(ttl_examples)
            mean_valid_accu = ttl_valid_accu/float(ttl_examples)
            print('\nvalid cost', mean_valid_cost)
            print('valid accu', mean_valid_accu)
            if best_valid_accu < mean_valid_accu:
                best_valid_accu = mean_valid_accu

            if es.continue_learning(valid_error=mean_valid_cost, epoch=epoch):
                print('epoch', epoch)
                print('best epoch last update:', es.best_epoch_last_update)
                print('best valid last update:', es.best_valid_last_update)
                print('best valid accuracy:', best_valid_accu)
            else:
                print('training done!')
                break


def train_with_trainobject():
    from tensorgraph.trainobject import train as mytrain
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        X_train, y_train, X_test, y_test = Cifar10(contrast_normalize=False, whiten=False)
        _, h, w, c = X_train.shape
        _, nclass = y_train.shape
        seq = model(nclass=nclass, h=h, w=w, c=c)

        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, nclass])

        y_train_sb = seq.train_fprop(X_ph)
        y_test_sb = seq.test_fprop(X_ph)
        train_cost_sb = entropy(y_ph, y_train_sb)
        optimizer = tf.train.AdamOptimizer(0.001)
        test_accu_sb = accuracy(y_ph, y_test_sb)

        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=100,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=64, randomize_split=False)


def train_with_VGG():
    from tensorgraph.trainobject import train as mytrain
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        X_train, y_train, X_test, y_test = Cifar10(contrast_normalize=False, whiten=False)
        _, h, w, c = X_train.shape
        _, nclass = y_train.shape
        print('X max', np.max(X_train))
        print('X min', np.min(X_train))
        from tensorgraph.layers import VGG19
        seq = tg.Sequential()
        layer = VGG19(input_channels=c, input_shape=(h,w))
        seq.add(layer)
        seq.add(Flatten())
        seq.add(Linear(512,nclass))
        seq.add(Softmax())
        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, nclass])

        y_train_sb = seq.train_fprop(X_ph)
        y_test_sb = seq.test_fprop(X_ph)
        train_cost_sb = entropy(y_ph, y_train_sb)
        optimizer = tf.train.AdamOptimizer(0.001)
        test_accu_sb = accuracy(y_ph, y_test_sb)

        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=100,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=64, randomize_split=False)


def train_with_Resnet():
    from tensorgraph.trainobject import train as mytrain
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        X_train, y_train, X_test, y_test = Cifar10(contrast_normalize=False, whiten=False)
        _, h, w, c = X_train.shape
        _, nclass = y_train.shape
        print('X max', np.max(X_train))
        print('X min', np.min(X_train))
        seq = tg.Sequential()
        id1 = IdentityBlock(input_channels=c, input_shape=(h,w), nlayers=4, filters=[32, 64])
        seq.add(id1)
        trans1 = TransitionLayer(input_channels=id1.output_channels, input_shape=id1.output_shape)
        seq.add(trans1)

        id2 = IdentityBlock(input_channels=trans1.output_channels, input_shape=trans1.output_shape,
                            nlayers=4, filters=[64, 128])
        seq.add(id2)
        trans2 = TransitionLayer(input_channels=id2.output_channels, input_shape=id2.output_shape)
        seq.add(trans2)
        seq.add(Flatten())
        ldim = trans2.output_channels * np.prod(trans2.output_shape)
        seq.add(Linear(ldim,nclass))
        seq.add(Softmax())

        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, nclass])

        y_train_sb = seq.train_fprop(X_ph)
        y_test_sb = seq.test_fprop(X_ph)
        train_cost_sb = entropy(y_ph, y_train_sb)
        optimizer = tf.train.AdamOptimizer(0.001)
        test_accu_sb = accuracy(y_ph, y_test_sb)

        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=100,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=64, randomize_split=False)


def train_with_Densenet():
    from tensorgraph.trainobject import train as mytrain
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        X_train, y_train, X_test, y_test = Cifar10(contrast_normalize=False, whiten=False)
        _, h, w, c = X_train.shape
        _, nclass = y_train.shape
        print('X max', np.max(X_train))
        print('X min', np.min(X_train))
        seq = tg.Sequential()
        dense = DenseNet(input_channels=c, input_shape=(h,w), ndense=3, growth_rate=4, nlayer1blk=4)
        seq.add(dense)
        seq.add(Flatten())
        ldim = dense.output_channels
        seq.add(Linear(ldim,nclass))
        seq.add(Softmax())

        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, nclass])

        y_train_sb = seq.train_fprop(X_ph)
        y_test_sb = seq.test_fprop(X_ph)
        train_cost_sb = entropy(y_ph, y_train_sb)
        optimizer = tf.train.AdamOptimizer(0.001)
        test_accu_sb = accuracy(y_ph, y_test_sb)

        print(tf.global_variables())
        print('..total number of global variables: {}'.format(len(tf.global_variables())))
        count = 0
        for var in tf.global_variables():
            count += int(np.prod(var.get_shape()))
        print('..total number of global parameters: {}'.format(count))

        mytrain(session=sess,
                feed_dict={X_ph:X_train, y_ph:y_train},
                train_cost_sb=train_cost_sb,
                valid_cost_sb=-test_accu_sb,
                optimizer=optimizer,
                epoch_look_back=5, max_epoch=100,
                percent_decrease=0, train_valid_ratio=[5,1],
                batchsize=64, randomize_split=False)


if __name__ == '__main__':
    # train()
    # train_with_trainobject()
    # train_with_VGG()
    # train_with_Resnet()
    train_with_Densenet()
