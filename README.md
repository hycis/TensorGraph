# TensorGraph
A graph based on TensorFlow

As deep learning becomes more and more common and the architectures becoming more
and more complicated, it seems that we need some easy to use framework to quickly
build some models and that's why TensorGraph is born. It's a very simple and easy
to use framework, but it allows you to build all kinds of imaginable models. The
idea is simple,


## transfer learning model

## hierachical softmax

## some monster model you can imagine

<img src="graph.png" height="250">

example
```python
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
```
