
from tensorgraph.node import StartNode, HiddenNode, EndNode
import tensorflow as tf
from tensorgraph.layers.linear import Linear
from tensorgraph.layers.activation import RELU, Softmax
from tensorgraph.layers.merge import Concat, Mean, Sum
from tensorgraph.graph import Graph
import numpy as np
from tensorgraph.data_iterator import SequentialIterator

## params
x_dim = 50
component_dim = 100
batchsize = 32
learning_rate = 0.01


x_ph = tf.placeholder('float32', [None, x_dim])
# the three components
y1_ph = tf.placeholder('float32', [None, component_dim])
y2_ph = tf.placeholder('float32', [None, component_dim])
y3_ph = tf.placeholder('float32', [None, component_dim])

# define the graph model structure
start = StartNode(input_vars=[x_ph])

h1 = HiddenNode(prev=[start], layers=[Linear(x_dim, component_dim), Softmax()])
h2 = HiddenNode(prev=[h1], layers=[Linear(component_dim, component_dim), Softmax()])
h3 = HiddenNode(prev=[h2], layers=[Linear(component_dim, component_dim), Softmax()])


e1 = EndNode(prev=[h1], input_merge_mode=Sum())
e2 = EndNode(prev=[h1, h2], input_merge_mode=Sum())
e3 = EndNode(prev=[h1, h2, h3], input_merge_mode=Sum())

graph = Graph(start=[start], end=[e1, e2, e3])

o1, o2, o3 = graph.train_fprop()

o1_mse = tf.reduce_mean((y1_ph - o1)**2)
o2_mse = tf.reduce_mean((y2_ph - o2)**2)
o3_mse = tf.reduce_mean((y3_ph - o3)**2)
mse = o1_mse + o2_mse + o3_mse
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)

X = np.random.rand(1000, x_dim)
Y1 = np.random.rand(1000, component_dim)
Y2 = np.random.rand(1000, component_dim)
Y3 = np.random.rand(1000, component_dim)

data = SequentialIterator(X, Y1, Y2, Y3, batchsize=batchsize)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    i = 0
    for x_batch, y1_batch, y2_batch, y3_batch in data:
        print(i)
        i += 1
        sess.run(optimizer, feed_dict={x_ph:x_batch, y1_ph:y1_batch, y2_ph:y2_batch, y3_ph:y3_batch})
