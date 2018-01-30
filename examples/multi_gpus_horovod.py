
import tensorgraph as tg
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import cifar10_allcnn
from tensorflow.python.framework import ops

hvd.init()

def cifar10(create_tfrecords=True, batch_size=32):
    tfrecords = tg.utils.MakeTFRecords()
    tfpath_train = './cifar10_train.tfrecords'
    tfpath_test = './cifar10_test.tfrecords'
    if create_tfrecords:
        X_train, y_train, X_test, y_test = tg.dataset.Cifar10()
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_train, 'y':y_train}, save_path=tfpath_train)
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_test, 'y':y_test}, save_path=tfpath_test)


    nr_train = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath_train],
                                              data_shapes={'X':[32,32,3], 'y':[10]},
                                              batch_size=batch_size)
    nr_test = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath_test],
                                              data_shapes={'X':[32,32,3], 'y':[10]},
                                              batch_size=batch_size)

    n_train = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath_train))
    n_test = sum(1 for _ in tf.python_io.tf_record_iterator(tfpath_test))
    return dict(nr_train), n_train, dict(nr_test), n_test


def train():
    graph = tf.Graph()
    with graph.as_default():
        batch_size = 100
        nr_train, n_train, nr_test, n_test = cifar10(create_tfrecords=True, batch_size=batch_size)
        seq = cifar10_allcnn.model(nclass=10, h=32, w=32, c=3)

        y_train_sb = seq.train_fprop(nr_train['X'])
        y_test_sb = seq.test_fprop(nr_test['X'])

        loss_train_sb = tg.cost.mse(y_train_sb, nr_train['y'])
        accu_train_sb = tg.cost.accuracy(y_train_sb, nr_train['y'])
        accu_test_sb = tg.cost.accuracy(y_test_sb, nr_test['y'])

        opt = tf.train.RMSPropOptimizer(0.001)
        opt = hvd.DistributedOptimizer(opt)

        # required for BatchNormalization layer
        update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
        with ops.control_dependencies(update_ops):
            train_op = opt.minimize(loss_train_sb)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        bcast = hvd.broadcast_global_variables(0)

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(graph=graph, config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        bcast.run()

        for epoch in range(100):
            pbar = tg.ProgressBar(n_train)
            ttl_train_loss = 0
            for i in range(0, n_train, batch_size):
                pbar.update(i)
                _, loss_train = sess.run([train_op, loss_train_sb])
                ttl_train_loss += loss_train * batch_size
            pbar.update(n_train)
            ttl_train_loss /= n_train
            print('epoch {}, train loss {}'.format(epoch, ttl_train_loss))

            pbar = tg.ProgressBar(n_test)
            ttl_test_loss = 0
            for i in range(0, n_test, batch_size):
                pbar.update(i)
                loss_test = sess.run(accu_test_sb)
                ttl_test_loss += loss_test * batch_size
            pbar.update(n_test)
            ttl_test_loss /= n_test
            print('epoch {}, test accuracy {}'.format(epoch, ttl_test_loss))


        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()
