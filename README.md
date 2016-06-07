# TensorGraph
TensorGraph is a framework for building any imaginable models based on TensorFlow.


-----
TensorGraph has three types of nodes

1. StartNode : for inputs to the graph
2. HiddenNode : for putting sequential layers inside
3. EndNode : for getting outputs from the model

The graph always starts with `StartNode` and ends with `EndNode`. Below shows an
[example](../examples/example.py) of building a tensor graph.

-----
### Graph Example

<img src="draw/graph.png" height="250">

First define the `StartNode`
```python
y1_dim = 50
y2_dim = 100
batchsize = 32
learning_rate = 0.01

y1 = tf.placeholder('float32', [None, y1_dim])
y2 = tf.placeholder('float32', [None, y2_dim])
s1 = StartNode(input_vars=[y1])
s2 = StartNode(input_vars=[y2])
```
Then define the `HiddenNode`
```python
h1 = HiddenNode(prev=[s1, s2],
                input_merge_mode=Concat(),
                layers=[Linear(y1_dim+y2_dim, y2_dim), RELU()])
h2 = HiddenNode(prev=[start2],
                layers=[Linear(y2_dim, y2_dim), RELU()])
h3 = HiddenNode(prev=[h1, h2],
                input_merge_mode=Sum(),
                layers=[Linear(y2_dim, y1_dim), RELU()])
```
Then define the `EndNode`
```python
e1 = EndNode(prev=[h3])
e2 = EndNode(prev=[h2])
```
Finally build the graph by putting `StartNodes` and `EndNodes` into `Graph`
```python
graph = Graph(start=[start1, start2], end=[e1, e2])
```
Run train forward propogation to get symbolic output from train mode, the number
of outputs from `graph.train_fprop` is the same as the number of `EndNodes` put
into `Graph`
```python
o1, o2 = graph.train_fprop()
```
Finally build an optimizer to optimizer the objective function
```python
o1_mse = tf.reduce_mean((y1 - o1)**2)
o2_mse = tf.reduce_mean((y2 - o2)**2)
mse = o1_mse + o2_mse
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
```
-----
### Hierachical Softmax Example
Below is another example for building a more powerful hierachical softmax

<img src="draw/hsoftmax.png" height="250">

```python
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
```
