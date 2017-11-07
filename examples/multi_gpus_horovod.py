
import tensorgraph as tg
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import cifar10_allcnn

hvd.init()

def cifar10(create_tfrecords=True, batch_size=32):
    tfrecords = tg.utils.MakeTFRecords()
    if create_tfrecords:
        X_train, y_train, X_test, y_test = tg.dataset.Cifar10()
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X_train, 'y':y_train}, save_path='./cifar10.tfrecords')
    names_records = tfrecords.read_and_decode(tfrecords_filename_list=['./cifar10.tfrecords'],
                                              data_shapes={'X':[32,32,3], 'y':[10]},
                                              batch_size=batch_size)
    return dict(names_records)


def train():
    graph = tf.Graph()
    with graph.as_default():
        batch_size = 100
        names_records = cifar10(create_tfrecords=True, batch_size=batch_size)
        seq = cifar10_allcnn.model(nclass=10, h=32, w=32, c=3)
        y_train_sb = seq.train_fprop(names_records['X'])
        loss_train_sb = tg.cost.mse(y_train_sb, names_records['y'])
        accu_train_sb = tg.cost.accuracy(y_train_sb, names_records['y'])
        opt = tf.train.RMSPropOptimizer(0.01)
        opt = hvd.DistributedOptimizer(opt)
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

        n_exp = 50000
        for epoch in range(100):
            pbar = tg.ProgressBar(n_exp)
            ttl_train_loss = 0
            for i in range(0, n_exp, batch_size):
                pbar.update(i)
                _, loss_train = sess.run([train_op, loss_train_sb])
                ttl_train_loss += loss_train
            pbar.update(n_exp)
            ttl_train_loss /= n_exp
            print('epoch {}, train loss {}'.format(epoch, ttl_train_loss))

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()
