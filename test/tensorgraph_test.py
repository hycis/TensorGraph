
from tensorflow.python.layers.normalization import BatchNormalization as TFBatchNorm
from tensorgraphx.layers import Conv2D, BatchNormalization, RELU, Linear, Flatten, \
                               BaseModel, Sum
import tensorflow as tf
import numpy as np
import tensorgraphx as tg
from tensorflow.python.framework import ops
import os

class CBR(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, h, w, c):

        layers1 = []
        layers1.append(Conv2D(input_channels=c, num_filters=1, kernel_size=(2,2), stride=(1,1), padding='SAME'))
        layers1.append(BatchNormalization(input_shape=[h,w,1]))
        layers1.append(RELU())

        layers2 = []
        layers2.append(Conv2D(input_channels=c, num_filters=1, kernel_size=(2,2), stride=(1,1), padding='SAME'))
        layers2.append(BatchNormalization(input_shape=[h,w,1]))
        layers2.append(RELU())

        self.startnode = tg.StartNode(input_vars=[None])
        hn1 = tg.HiddenNode(prev=[self.startnode], layers=layers1)
        hn2 = tg.HiddenNode(prev=[self.startnode], layers=layers2)
        hn3 = tg.HiddenNode(prev=[hn1, hn2], input_merge_mode=Sum())
        self.endnode = tg.EndNode(prev=[hn3])


class TGModel(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, h, w, c, nclass):
        layers = []
        layers.append(CBR(h,w,c))
        layers.append(Flatten())
        layers.append(Linear(1*h*w, nclass))

        self.startnode = tg.StartNode(input_vars=[None])
        hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[hn])



def conv_layer(state_below, input_channels, num_filters, kernel_size, stride, padding):

    filter_shape = kernel_size + (input_channels, num_filters)

    _filter = tf.Variable(tf.random_normal(filter_shape, stddev=0.1), name='Conv2D_filter')
    _b = tf.Variable(tf.zeros([num_filters]), name='Conv2D_b')

    conv_out = tf.nn.conv2d(state_below, _filter, strides=(1,)+tuple(stride)+(1,),
                            padding=padding, data_format='NHWC')
    return tf.nn.bias_add(conv_out, _b)


def batchnorm(state_below, input_shape):
    bn = TFBatchNorm()
    bn.build(input_shape=[None] + list(input_shape))
    return bn.apply(state_below, training=True)



def linear(state_below, prev_dim=None, this_dim=None, W=None, b=None, stddev=0.1):
    W = tf.Variable(tf.random_normal([prev_dim, this_dim], stddev=0.1), name='Linear_W')
    b = tf.Variable(tf.zeros([this_dim]), name='Linear_b')
    return tf.matmul(state_below, W) + b



def data(n_exp, h, w, c, nclass, batch_size):
    tfrecords = tg.utils.MakeTFRecords()
    tfpath = './data.tf'
    if not os.path.exists(tfpath):
        X = np.random.rand(n_exp, h, w, c)
        y = np.random.rand(n_exp, nclass)
        tfrecords.make_tfrecords_from_arrs(data_records={'X':X, 'y':y}, save_path=tfpath)
    nr_train = tfrecords.read_and_decode(tfrecords_filename_list=[tfpath],
                                         data_shapes={'X':[h,w,c], 'y':[nclass]},
                                         batch_size=batch_size)
    return dict(nr_train)


def TFModel(state_below, h, w, c, nclass):
    state_below1 = conv_layer(state_below, input_channels=c, num_filters=1, kernel_size=(2,2), stride=(1,1), padding='SAME')
    state_below1 = batchnorm(state_below1, input_shape=[h,w,1])
    state_below1 = tf.nn.relu(state_below1)

    state_below2 = conv_layer(state_below, input_channels=c, num_filters=1, kernel_size=(2,2), stride=(1,1), padding='SAME')
    state_below2 = batchnorm(state_below2, input_shape=[h,w,1])
    state_below2 = tf.nn.relu(state_below2)

    state_below = state_below1 + state_below2
    state_below = tf.contrib.layers.flatten(state_below)
    state_below = linear(state_below, 1*h*w, nclass)

    return state_below


def train(n_exp, h, w, c, nclass, batch_size=100, tgmodel=True):
    graph = tf.Graph()
    with graph.as_default():
        # nr_train = data(n_exp, h, w, c, nclass, batch_size)
        X_data = np.random.rand(n_exp, h, w, c)
        y_data = np.random.rand(n_exp, nclass)
        data_iter = tg.SequentialIterator(X_data, y_data, batchsize=batch_size)

        X_ph = tf.placeholder('float32', [None, h, w, c])
        y_ph = tf.placeholder('float32', [None, nclass])

        if tgmodel:
            # tensorgraphx model
            print('..using graph model')
            seq = TGModel(h, w, c, nclass)
            y_train_sb = seq.train_fprop(X_ph)

        else:
            # tensorflow model
            print('..using tensorflow model')
            y_train_sb = TFModel(X_ph, h, w, c, nclass)

        loss_train_sb = tg.cost.mse(y_train_sb, y_ph)
        accu_train_sb = tg.cost.accuracy(y_train_sb, y_ph)

        opt = tf.train.RMSPropOptimizer(0.001)

        # required for BatchNormalization layer
        update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
        with ops.control_dependencies(update_ops):
            train_op = opt.minimize(loss_train_sb)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        sess.run(init_op)

        for epoch in range(2):
            pbar = tg.ProgressBar(n_exp)
            ttl_train_loss = 0
            # for i in range(0, n_exp, batch_size):
            i = 0
            for X_batch, y_batch in data_iter:
                pbar.update(i)
                i += len(X_batch)
                _, loss_train = sess.run([train_op, loss_train_sb],
                                          feed_dict={X_ph:X_batch, y_ph:y_batch})
                ttl_train_loss += loss_train * batch_size
            pbar.update(n_exp)
            ttl_train_loss /= n_exp
            print('epoch {}, train loss {}'.format(epoch, ttl_train_loss))


def test_compare_total_nodes():
    h, w, c, nclass = 20, 20, 5, 2
    X_ph = tf.placeholder('float32', [None, h, w, c])
    with tf.name_scope('tgmodel'):
        seq = TGModel(h, w, c, nclass)
        y_train_sb = seq.train_fprop(X_ph)
        num_tg_nodes = [x for x in tf.get_default_graph().get_operations() if x.name.startswith('tgmodel/')]
        print('num tg nodes:', len(num_tg_nodes))
    with tf.name_scope('tfmodel'):
        y_train_sb = TFModel(X_ph, h, w, c, nclass)
        num_tf_nodes = [x for x in tf.get_default_graph().get_operations() if x.name.startswith('tfmodel/')]
        print('num tf nodes:', len(num_tf_nodes))
    assert len(num_tg_nodes) == len(num_tf_nodes)


def test_models():
    train(n_exp=10, h=20, w=20, c=5, nclass=2, batch_size=1, tgmodel=False)
    train(n_exp=10, h=20, w=20, c=5, nclass=2, batch_size=1, tgmodel=True)



if __name__ == '__main__':
    test_models()
    test_compare_total_nodes()
