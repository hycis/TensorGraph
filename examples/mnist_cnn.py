# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

from __future__ import division, print_function, absolute_import

from tensorgraph.layers import Conv2D, RELU, MaxPooling, LRN, Tanh, Dropout, \
                               Softmax, Flatten, Linear, BatchNormalization
from tensorgraph.utils import same
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy
from tensorgraph.dataset import Mnist
from tensorflow.python.framework import ops

def model():
    with tf.name_scope('MnistCNN'):
        seq = tg.Sequential()
        seq.add(Conv2D(input_channels=1, num_filters=32, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        h, w = same(in_height=28, in_width=28, stride=(1,1), kernel_size=(3,3))
        seq.add(BatchNormalization(input_shape=[h,w,32]))
        seq.add(RELU())

        seq.add(MaxPooling(poolsize=(2, 2), stride=(2,2), padding='SAME'))
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(2,2))
        seq.add(LRN())

        seq.add(Conv2D(input_channels=32, num_filters=64, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        seq.add(BatchNormalization(input_shape=[h,w,64]))
        seq.add(RELU())

        seq.add(MaxPooling(poolsize=(2, 2), stride=(2,2), padding='SAME'))
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(2,2))
        seq.add(LRN())
        seq.add(Flatten())
        seq.add(Linear(int(h*w*64), 128))
        seq.add(BatchNormalization(input_shape=[128]))
        seq.add(Tanh())
        seq.add(Dropout(0.8))
        seq.add(Linear(128, 256))
        seq.add(BatchNormalization(input_shape=[256]))
        seq.add(Tanh())
        seq.add(Dropout(0.8))
        seq.add(Linear(256, 10))
        seq.add(Softmax())
    return seq


def train():
    learning_rate = 0.001
    batchsize = 32

    max_epoch = 300
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)

    seq = model()
    X_train, y_train, X_test, y_test = Mnist(flatten=False, onehot=True, binary=True, datadir='.')
    iter_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
    iter_test = tg.SequentialIterator(X_test, y_test, batchsize=batchsize)

    X_ph = tf.placeholder('float32', [None, 28, 28, 1])
    y_ph = tf.placeholder('float32', [None, 10])

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
    with tf.Session() as sess:
        seq = model()
        X_train, y_train, X_test, y_test = Mnist(flatten=False, onehot=True, binary=True, datadir='.')
        X_ph = tf.placeholder('float32', [None, 28, 28, 1])
        y_ph = tf.placeholder('float32', [None, 10])
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


if __name__ == '__main__':
    # train()
    train_with_trainobject()
