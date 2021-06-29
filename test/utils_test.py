<<<<<<< HEAD
from tensorgraph.utils import MakeTFRecords,MakeTFRecords_tfdata
=======
from tensorgraphx.utils import MakeTFRecords
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
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

<<<<<<< HEAD
=======

>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
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

<<<<<<< HEAD
def test_make_tfrecords_tfdata():
    tfrecords = MakeTFRecords_tfdata()
    data_records = {'X':np.random.rand(100,50,30), 'y':np.random.rand(100,10),'name':['a']*20+['b']*20+['c']*20+['d']*40}
    save_path = './arr.tfrecords'
    tfrecords.make_tfrecords_from_arrs(data_records, save_path,[np.float32,np.float32,str])
    print('successfully created tfrecords.')

def test_fetch_queue_tfrecords_tfdata():
    tfrecords = MakeTFRecords_tfdata()
    tfrecords_filename = './arr.tfrecords'

    data_records = {'X':np.random.rand(100,50,30), 'y':np.random.rand(100,10),'name':['a']*20+['b']*20+['c']*20+['d']*40}
    tfrecords.make_tfrecords_from_arrs(data_records, tfrecords_filename,[np.float32,np.float32,str])

    num_epochs = 3
    batch_size = 10
    n_train = sum(1 for _ in tf.python_io.tf_record_iterator(tfrecords_filename))
    it,element = tfrecords.read_tensor_from_tfrecords(n_train,
                                                      tfrecords_filenames=[tfrecords_filename],
                                                      data_shape={'X':[50,30], 'y':[10],'name':None},
                                                      dtypes=[tf.float32,tf.float32,str],
                                                      batch_size=batch_size,num_epochs=num_epochs,phase='training')
    with tf.Session() as sess:
        sess.run(it.initializer)
        batches = 100/batch_size
        keys = element.keys()
        for i in range(num_epochs):
            for j in range(int(batches)):
                arrs =sess.run([element[k] for k in keys])
                for index, key in enumerate(keys):
                    print (key, arrs[index].shape)
                    print('\n')

=======
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5

if __name__ == '__main__':
    test_make_tfrecords()
    test_fetch_queue_tfrecords()
<<<<<<< HEAD
    test_make_tfrecords_tfdata()
    test_fetch_queue_tfrecords_tfdata()
=======
>>>>>>> e55a706e1467da7b7c54b6d04055aba847f5a2b5
