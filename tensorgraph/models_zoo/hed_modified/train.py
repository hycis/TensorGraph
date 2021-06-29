
import tensorflow as tf
import numpy as np
from ...utils import split_arr
from ...data_iterator import SequentialIterator
from ...cost import mean_dice, inv_dice



D, H, W = 16, 64, 64
X_train = np.random.rand(6, D, H, W, 1)
y_train = np.random.rand(6, D, H, W, 1)


def train(seq, X_ph, y_ph, X_train, y_train):
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb  = seq.test_fprop(X_ph)

    train_dice_tf    = mean_dice(y_ph, y_train_sb)
    valid_dice_tf    = mean_dice(y_ph, y_test_sb)
    train_invLoss_tf = inv_dice(y_ph, y_train_sb)
    valid_invLoss_tf = inv_dice(y_ph, y_test_sb)

    train_cost_tf     = 1.0 * train_dice_tf + 0.0 * train_invLoss_tf
    valid_cost_tf     = 1.0 * valid_dice_tf + 0.0 * valid_invLoss_tf
    reg_loss_tf       = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    train_cost_reg_tf = tf.add_n([train_cost_tf] + reg_loss_tf)
    valid_cost_reg_tf = tf.add_n([valid_cost_tf] + reg_loss_tf)

    total_epochs = 1
    print_period = 1
    batchsize = 1
    lr = 1e-3
    decay_steps = 380
    t_mul       = 1.027
    m_mul       = 0.987
    min_ratio   = 0.081

    global_step_tf = tf.Variable(0, trainable=False)
    decayed_lr_tf = tf.train.cosine_decay_restarts(lr,
                                                   global_step_tf,
                                                   decay_steps,
                                                   t_mul, m_mul, min_ratio)
    optimizer = tf.train.AdamOptimizer(learning_rate=decayed_lr_tf,
                                       epsilon=10**-6)
    with tf.variable_scope('AdamOptimizer'):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(train_cost_reg_tf,
                                          global_step_tf)

    train_arrs = []
    valid_arrs = []
    phs = []

    feed_dict={X_ph:X_train, y_ph:y_train}

    for ph, arr in feed_dict.items():
        train_arr, valid_arr = split_arr(arr, [5,1], randomize=False)
        phs.append(ph)
        train_arrs.append(train_arr)
        valid_arrs.append(valid_arr)

    iter_train = SequentialIterator(*train_arrs, batchsize=batchsize)
    iter_valid = SequentialIterator(*valid_arrs, batchsize=batchsize)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    Holder_trainloss = [0] * total_epochs
    Holder_validloss = [0] * total_epochs

    for i in range(total_epochs):
        j = 0
        for batches in iter_train:
            j += 1
            fd = dict(zip(phs, batches))
            _, loss, invloss = sess.run([train_op, train_cost_reg_tf, train_invLoss_tf],
                                        feed_dict=fd)
            Holder_trainloss[i] += loss

            if j % print_period == 0:
                print("Epoch i: %d, j: %d, Training loss: %.3f" % (i, j, loss))

        Holder_trainloss[i] /= j
        print("Completed training all batches in epoch %d. Performing validation..." % i)

        k = 0
        for batches in iter_valid:
            k += 1
            fd = dict(zip(phs, batches))
            valid_loss, valid_invloss = sess.run([valid_cost_reg_tf, valid_invLoss_tf],
                                                 feed_dict=fd)
            Holder_validloss[i] += valid_loss
            print("Validation loss is %.3f" % (valid_loss))

        Holder_validloss[i] /= k

        print("Avg train loss for epoch %d: %.3f" % (i, Holder_trainloss[i]))
        print("Avg valid loss for epoch %d: %.3f" % (i, Holder_validloss[i]))
