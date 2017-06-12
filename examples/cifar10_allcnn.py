# -*- coding: utf-8 -*-

"""
From the paper: STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
https://arxiv.org/pdf/1412.6806.pdf
"""

from __future__ import division, print_function, absolute_import

from tensorgraph.layers import Conv2D, RELU, MaxPooling, LRN, Tanh, Dropout, \
                               Softmax, Flatten, Linear, TFBatchNormalization, AvgPooling, \
                               Lambda
from tensorgraph.utils import same, valid
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy
from tensorgraph.dataset import Mnist, Cifar10

def model(nclass, h, w, c):
    with tf.name_scope('Cifar10AllCNN'):
        seq = tg.Sequential()
        seq.add(Conv2D(input_channels=c, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b1'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b3'))
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=96, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b5'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b7'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

        seq.add(Conv2D(input_channels=192, num_filters=192, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
        seq.add(Dropout(0.5))

        seq.add(Conv2D(input_channels=192, num_filters=nclass, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        seq.add(RELU())
        seq.add(TFBatchNormalization(name='b9'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))

        seq.add(AvgPooling(poolsize=(h, w), stride=(1,1), padding='VALID'))
        seq.add(Flatten())
        seq.add(Softmax())
    return seq


if __name__ == '__main__':

    learning_rate = 0.001
    batchsize = 64

    max_epoch = 300
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=None,
                         percent_decrease=0)



    X_train, y_train, X_test, y_test = Cifar10(contrast_normalize=False, whiten=False)
    # X_train, y_train, X_test, y_test = Mnist(flatten=False, onehot=True, binary=True, datadir='.')
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

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_sb)

    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
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
                _, train_cost = sess.run([optimizer,train_cost_sb] , feed_dict=feed_dict)
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
