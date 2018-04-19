from tensorgraphx.utils import MakeTFRecords
import numpy as np
import tensorflow as tf


def test_make_tfrecords():
    tfrecords = MakeTFRecords()
    data_records = {'X':np.random.rand(100,50,30), 'y':np.random.rand(100,10)}
    save_path = './arr.tfrecords'
    tfrecords.make_tfrecords_from_arrs(data_records, save_path)
    arrs = tfrecords.read_arrs_from_tfrecords(save_path, data_shapes={'X':[50,30], 'y':[10]})
    for records in arrs:
        for record in records:
            print(record.shape)
        print('\n')


def test_fetch_queue_tfrecords():
    tfrecords = MakeTFRecords()
    tfrecords_filename = './arr.tfrecords'
    names_records = tfrecords.read_and_decode([tfrecords_filename],
                                              batch_size=1,
                                              data_shapes={'X':[50,30], 'y':[10]})
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(3):
            for name, record in names_records:
                arr = sess.run(record)
                print(name)
                print(arr.shape)
                print('\n')
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test_make_tfrecords()
    test_fetch_queue_tfrecords()
