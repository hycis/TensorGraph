# TensorGraph
TensorGraph is a framework for building any imaginable models based on TensorFlow.



TensorGraph has three types of nodes
1. StartNode : for inputs to the graph
2. HiddenNode : for putting layers and models inside
3. EndNode : for getting outputs from the model

The graph always starts with `StartNode` and ends with `EndNode`. Below shows an
[example](../examples/example.py) of building a tensor graph.

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
