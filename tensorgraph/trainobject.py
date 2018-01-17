
from .stopper import EarlyStopper
from .progbar import ProgressBar
from .utils import split_arr
from .data_iterator import SequentialIterator
from tensorflow.python.framework import ops
import tensorflow as tf
import logging
logging.basicConfig(format='%(module)s.%(funcName)s %(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def train(session, feed_dict, train_cost_sb, valid_cost_sb, optimizer, epoch_look_back=5,
          max_epoch=100, percent_decrease=0, train_valid_ratio=[5,1], batchsize=64,
          randomize_split=False):

    train_arrs = []
    valid_arrs = []
    phs = []
    for ph, arr in feed_dict.items():
        train_arr, valid_arr = split_arr(arr, train_valid_ratio, randomize=randomize_split)
        phs.append(ph)
        train_arrs.append(train_arr)
        valid_arrs.append(valid_arr)

    iter_train = SequentialIterator(*train_arrs, batchsize=batchsize)
    iter_valid = SequentialIterator(*valid_arrs, batchsize=batchsize)

    es = EarlyStopper(max_epoch, epoch_look_back, percent_decrease)

    # required for BatchNormalization layer
    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    with ops.control_dependencies(update_ops):
        train_op = optimizer.minimize(train_cost_sb)

    init = tf.global_variables_initializer()
    session.run(init)

    epoch = 0
    while True:
        epoch += 1
        ##############################[ Training ]##############################
        print('\n')
        logger.info('<<<<<[ epoch: {} ]>>>>>'.format(epoch))
        logger.info('..training')
        pbar = ProgressBar(len(iter_train))
        ttl_exp = 0
        mean_train_cost = 0
        for batches in iter_train:
            fd = dict(zip(phs, batches))
            train_cost, _ = session.run([train_cost_sb, train_op], feed_dict=fd)
            mean_train_cost += train_cost * len(batches[0])
            ttl_exp += len(batches[0])
            pbar.update(ttl_exp)

        print('')
        mean_train_cost /= ttl_exp
        logger.info('..average train cost: {}'.format(mean_train_cost))

        ##############################[ Validating ]############################
        logger.info('..validating')
        pbar = ProgressBar(len(iter_valid))
        ttl_exp = 0
        mean_valid_cost = 0
        for batches in iter_valid:
            fd = dict(zip(phs, batches))
            valid_cost = session.run(valid_cost_sb, feed_dict=fd)
            mean_valid_cost += valid_cost * len(batches[0])
            ttl_exp += len(batches[0])
            pbar.update(ttl_exp)

        print('')
        mean_valid_cost /= ttl_exp
        logger.info('..average valid cost: {}'.format(mean_valid_cost))

        if es.continue_learning(mean_valid_cost, epoch=epoch):
            logger.info('best epoch last update: {}'.format(es.best_epoch_last_update))
            logger.info('best valid last update: {}'.format(es.best_valid_last_update))
        else:
            logger.info('training done!')
            break
