
from tensorgraph.node import StartNode, HiddenNode, EndNode
import tensorflow as tf
from tensorgraph.layers.linear import Linear
from tensorgraph.layers.activation import RELU
from tensorgraph.layers.misc import Concat, Mean, Sum
from tensorgraph.graph import Graph
import numpy as np
from tensorgraph.data_iterator import SequentialIterator


y1_dim = 50
y2_dim = 100
batchsize = 32
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


Y1 = np.random.rand(1000, 50)
Y2 = np.random.rand(1000, 100)
data = SequentialIterator(Y1, Y2, batchsize=batchsize)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for y1_batch, y2_batch in data:
        sess.run([o1, o2], feed_dict={y1:y1_batch, y2:y2_batch})
        h1w = sess.run(h1.layers[0].W, feed_dict={y1:y1_batch, y2:y2_batch})
        h2w = sess.run(h2.layers[0].W, feed_dict={y1:y1_batch, y2:y2_batch})
        h3w = sess.run(h3.layers[0].W, feed_dict={y1:y1_batch, y2:y2_batch})
        print 'before optimize'
        print 'h1w', np.mean(h1w)
        print 'h2w', np.mean(h2w)
        print 'h3w', np.mean(h3w)
        sess.run(optimizer, feed_dict={y1:y1_batch, y2:y2_batch})

        print 'after optimize'
        h1w = sess.run(h1.layers[0].W, feed_dict={y1:y1_batch, y2:y2_batch})
        h2w = sess.run(h2.layers[0].W, feed_dict={y1:y1_batch, y2:y2_batch})
        h3w = sess.run(h3.layers[0].W, feed_dict={y1:y1_batch, y2:y2_batch})
        print 'h1w', np.mean(h1w)
        print 'h2w', np.mean(h2w)
        print 'h3w', np.mean(h3w)
