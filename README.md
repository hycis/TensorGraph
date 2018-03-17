[![Build Status](https://travis-ci.org/hycis/TensorGraph.svg?branch=master)](https://travis-ci.org/hycis/TensorGraph)

# TensorGraph - Simplicity is Beauty
TensorGraph is a simple, lean, and clean framework on TensorFlow for building any imaginable models.

As deep learning becomes more and more common and the architectures becoming more
and more complicated, it seems that we need some easy to use framework to quickly
build these models and that's what TensorGraph is designed for. It's a very simple
framework that adds a very thin layer above tensorflow. It is for more advanced
users who want to have more control and flexibility over his model building and
who wants efficiency at the same time.

-----
### Target Audience
TensorGraph is targeted more at intermediate to advance users who feel keras or
other packages is having too much restrictions and too much black box on model
building, and someone who don't want to rewrite the standard layers in tensorflow
constantly. Also for enterprise users who want to share deep learning models
easily between teams.

-----
### Install

First you need to install [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)

To install tensorgraph simply do via pip
```bash
sudo pip install tensorgraph
```
or for bleeding edge version do
```bash
sudo pip install --upgrade git+https://github.com/hycis/TensorGraph.git@master
```
or simply clone and add to `PYTHONPATH`.
```bash
git clone https://github.com/hycis/TensorGraph.git
export PYTHONPATH=/path/to/TensorGraph:$PYTHONPATH
```
in order for the install to persist via export `PYTHONPATH`. Add `PYTHONPATH=/path/to/TensorGraph:$PYTHONPATH` to your `.bashrc` for linux or
`.bash_profile` for mac. While this method works, you will have to ensure that
all the dependencies in [setup.py](setup.py) are installed.

-----
### Everything in TensorGraph is about Layers
Everything in TensorGraph is about layers. A model such as VGG or Resnet can be a layer. An identity block from Resnet or a dense block from Densenet can be a layer as well. Building models in TensorGraph is same as building a toy with lego. For example you can create a new model (layer) by subclass the `BaseModel` layer and use `DenseBlock` layer inside your `ModelA` layer.

```python
from tensorgraph.layers import DenseBlock, BaseModel, Flatten, Linear, Softmax
import tensorgraph as tg

class ModelA(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self):
        layers = []
        layers.append(DenseBlock())
        layers.append(Flatten())
        layers.append(Linear())
        layers.append(Softmax())
        self.startnode = tg.StartNode(input_vars=[None])
        hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[hn])
```

if someone wants to use your `ModelA` in his `ModelB`, he can easily do this
```python
class ModelB(BaseModel):
    @BaseModel.init_name_scope
    def __int__(self):
        layers = []
        layers.append(ModelA())
        layers.append(Linear())
        layers.append(Softmax())
        self.startnode = tg.StartNode(input_vars=[None])
        hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[hn])
```

creating a layer only created all the `Variables`. To connect the `Variables` into a graph, you can do a `train_fprop(X)` or `test_fprop(X)` to create the tensorflow graph. By abstracting `Variable` creation away from linking the `Variable` nodes into graph prevent the problem of certain tensorflow layers that always reinitialise its weights when it's called, example the [`tf.nn.batch_normalization`](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization) layer. Also having a separate channel for training and testing is to cater to layers with different training and testing behaviours such as batchnorm and dropout.

```python
modelb = ModelB()
X_ph = tf.placeholder()
y_train = modelb.train_fprop(X_ph)
y_test = modelb.test_fprop(X_ph)
```

checkout some well known models in TensorGraph
1. [VGG16 code](tensorgraph/layers/backbones.py#L37) and [VGG19 code](tensorgraph/layers/backbones.py#L125) - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
2. [DenseNet code](tensorgraph/layers/backbones.py#L477) - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
3. [ResNet code](tensorgraph/layers/backbones.py#L225) - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
4. [Unet code](tensorgraph/layers/backbones.py#L531) - [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

-----
### TensorGraph on Multiple GPUS
To use tensorgraph on multiple gpus, you can easily integrate it with [horovod](https://github.com/uber/horovod).

```python
import horovod.tensorflow as hvd
from tensorflow.python.framework import ops
import tensorflow as tf
hvd.init()

# tensorgraph model derived previously
modelb = ModelB()
X_ph = tf.placeholder()
y_ph = tf.placeholder()
y_train = modelb.train_fprop(X_ph)
y_test = modelb.test_fprop(X_ph)

train_cost = mse(y_train, y_ph)
test_cost = mse(y_test, y_ph)

opt = tf.train.RMSPropOptimizer(0.001)
opt = hvd.DistributedOptimizer(opt)

# required for BatchNormalization layer
update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
with ops.control_dependencies(update_ops):
    train_op = opt.minimize(train_cost)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
bcast = hvd.broadcast_global_variables(0)

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())

with tf.Session(graph=graph, config=config) as sess:
    sess.run(init_op)
    bcast.run()

    # training model
    for epoch in range(100):
        for X,y in train_data:
            _, loss_train = sess.run([train_op, train_cost], feed_dict={X_ph:X, y_ph:y})
```

for a full example on [tensorgraph on horovod](./examples/multi_gpus_horovod.py)

-----
### How TensorGraph Works?
In TensorGraph, we defined three types of nodes

1. StartNode : for inputs to the graph
2. HiddenNode : for putting sequential layers inside
3. EndNode : for getting outputs from the model

We put all the sequential layers into a `HiddenNode`, and connect the hidden nodes
together to build the architecture that you want. The graph always
starts with `StartNode` and ends with `EndNode`. The `StartNode` is where you place
your starting point, it can be a `placeholder`, a symbolic output from another graph,
or data output from `tfrecords`. `EndNode` is where you want to get an output from
the graph, where the output can be used to calculate loss or simply just a peek at the
outputs at that particular layer. Below shows an
[example](examples/example.py) of building a tensor graph.

-----
### Graph Example

<img src="draw/graph.png" height="250">

First define the `StartNode` for putting the input placeholder
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
Then define the `HiddenNode` for putting the sequential layers in each `HiddenNode`
```python
h1 = HiddenNode(prev=[s1, s2],
                input_merge_mode=Concat(),
                layers=[Linear(y1_dim+y2_dim, y2_dim), RELU()])
h2 = HiddenNode(prev=[s2],
                layers=[Linear(y2_dim, y2_dim), RELU()])
h3 = HiddenNode(prev=[h1, h2],
                input_merge_mode=Sum(),
                layers=[Linear(y2_dim, y1_dim), RELU()])
```
Then define the `EndNode`. `EndNode` is used to back-trace the graph to connect
the nodes together.
```python
e1 = EndNode(prev=[h3])
e2 = EndNode(prev=[h2])
```
Finally build the graph by putting `StartNodes` and `EndNodes` into `Graph`
```python
graph = Graph(start=[s1, s2], end=[e1, e2])
```
Run train forward propagation to get symbolic output from train mode. The number
of outputs from `graph.train_fprop` is the same as the number of `EndNodes` put
into `Graph`
```python
o1, o2 = graph.train_fprop()
```
Finally build an optimizer to optimize the objective function
```python
o1_mse = tf.reduce_mean((y1 - o1)**2)
o2_mse = tf.reduce_mean((y2 - o2)**2)
mse = o1_mse + o2_mse
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
```
-----
### Hierachical Softmax Example
Below is another example for building a more powerful [hierachical softmax](examples/hierachical_softmax.py)
whereby the lower hierachical softmax layer can be conditioned on all the upper
hierachical softmax layers.

<img src="draw/hsoftmax.png" height="250">

```python
## params
x_dim = 50
component_dim = 100
batchsize = 32
learning_rate = 0.01


x_ph = tf.placeholder('float32', [None, x_dim])
# the three hierachical level
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

-----
### Transfer Learning Example
Below is an example on transfer learning with bi-modality inputs and merge at
the middle layer with shared representation, in fact, TensorGraph can be used
to build any number of modalities for transfer learning.

<img src="draw/transferlearn.png" height="250">

```python
## params
x1_dim = 50
x2_dim = 100
shared_dim = 200
y_dim = 100
batchsize = 32
learning_rate = 0.01


x1_ph = tf.placeholder('float32', [None, x1_dim])
x2_ph = tf.placeholder('float32', [None, x2_dim])
y_ph = tf.placeholder('float32', [None, y_dim])

# define the graph model structure
s1 = StartNode(input_vars=[x1_ph])
s2 = StartNode(input_vars=[x2_ph])

h1 = HiddenNode(prev=[s1], layers=[Linear(x1_dim, shared_dim), RELU()])
h2 = HiddenNode(prev=[s2], layers=[Linear(x2_dim, shared_dim), RELU()])
h3 = HiddenNode(prev=[h1,h2], input_merge_mode=Sum(),
                layers=[Linear(shared_dim, y_dim), Softmax()])

e1 = EndNode(prev=[h3])

graph = Graph(start=[s1, s2], end=[e1])
o1, = graph.train_fprop()

mse = tf.reduce_mean((y_ph - o1)**2)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
```
