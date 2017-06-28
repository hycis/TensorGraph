
import tensorflow as tf
import numpy as np
from tensorgraph import Graph, StartNode, HiddenNode, EndNode
from tensorgraph.layers import Linear, RELU, Concat, Mean, Sum
from tensorgraph import ProgressBar, SequentialIterator


def model():
    y1_dim = 50
    y2_dim = 100

    learning_rate = 0.01

    y1 = tf.placeholder('float32', [None, y1_dim])
    y2 = tf.placeholder('float32', [None, y2_dim])
    start1 = StartNode(input_vars=[y1])
    start2 = StartNode(input_vars=[y2])

    h1 = HiddenNode(prev=[start1, start2],
                    input_merge_mode=Concat(),
                    layers=[Linear(y1_dim+y2_dim, y2_dim), RELU()])
    h2 = HiddenNode(prev=[start2],
                    layers=[Linear(y2_dim, y2_dim), RELU()])
    h3 = HiddenNode(prev=[h1, h2],
                    input_merge_mode=Sum(),
                    layers=[Linear(y2_dim, y1_dim), RELU()])
    e1 = EndNode(prev=[h3])
    e2 = EndNode(prev=[h2])


    graph = Graph(start=[start1, start2], end=[e1, e2])
    o1, o2 = graph.train_fprop()

    o1_mse = tf.reduce_mean((y1 - o1)**2)
    o2_mse = tf.reduce_mean((y2 - o2)**2)
    mse = o1_mse + o2_mse
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
    return y1, y2, o1, o2, optimizer


def train():
    batchsize = 32
    y1, y2, o1, o2, optimizer = model()
    Y1 = np.random.rand(100, 50)
    Y2 = np.random.rand(100, 100)
    data = SequentialIterator(Y1, Y2, batchsize=batchsize)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        for i in range(10):
            pbar = ProgressBar(target=len(data))
            n_exp = 0
            for y1_batch, y2_batch in data:
                sess.run([o1, o2], feed_dict={y1:y1_batch, y2:y2_batch})
                sess.run(optimizer, feed_dict={y1:y1_batch, y2:y2_batch})
                n_exp += len(y1_batch)
                pbar.update(n_exp)
            print('end')
        saver.save(sess, 'test.tf')


if __name__ == '__main__':
    train()
