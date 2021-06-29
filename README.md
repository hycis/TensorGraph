`master` [![Build Status](http://54.222.242.222:1010/buildStatus/icon?job=TensorGraph/master)](http://54.222.242.222:1010/job/TensorGraph/master)
`develop` [![Build Status](http://54.222.242.222:1010/buildStatus/icon?job=TensorGraph/develop)](http://54.222.242.222:1010/job/TensorGraph/develop)


# TensorGraph
TensorGraph is a simple, lean, and clean framework on TensorFlow for building any imaginable models.

As deep learning becomes more and more common and the architectures becoming more
and more complicated, it seems that we need some easy to use framework to quickly
build these models and that's what TensorGraph is designed for. It's a very simple
framework that adds a very thin layer above tensorflow. It is for more advanced
users who want to have more control and flexibility over his model building and
who wants efficiency at the same time.

-----
## Target Audience
TensorGraph is targeted more at intermediate to advance users who feel keras or
other packages is having too much restrictions and too much black box on model
building, and someone who don't want to rewrite the standard layers in tensorflow
constantly. Also for enterprise users who want to share deep learning models
easily between teams.

-----
## Install

First you need to install [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)

To install tensorgraph for bleeding edge version via pip
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
## Everything in TensorGraph is about Layers
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
## Type of Layers
There are three types of layers, `BaseLayer`, `BaseModel` and `Merge`.

### BaseLayer
`BaseLayer` is a low lying layer that wraps tensorflow codes directly, and define
the low level operations that we want the tensorflow to perform within a layer.
When implementing `BaseLayer` we need to implement `_train_fprop()` and `_test_fprop()`,
by default `_test_fprop()` calls `_train_fprop()`.

```python
class MyLayer(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self):
        ''' place all your variables and variables initialization here. '''
        pass

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        '''Define variables which requires input information from state_below,
           this is called during forward propagation
        '''
        pass

    def _train_fprop(self, state_below):
        '''
        your tensorflow operations for training,
        do not initialize variables here.
        '''
        pass

    def _test_fprop(self, state_below):
        '''
        your tensorflow operations for testing, do not initialize variables
        here. Defaults to _train_fprop.
        '''
        pass
```

To use `BaseLayer`, we can initialize the `Variables` inside ```__init__``` and/or
```__init_var__(self, state_below)``` if our layer requires information from the
layer below.


### BaseModel
`BaseModel` is a higher level layer that can be made up of BaseLayers and
BaseModels. For BaseModel, a default implementation of `_train_fprop`
and `_test_fprop` has been done for a single `StartNode` and single `EndNode`
Graph, to use this default implementation, we have to define `self.startnode`
and `self.endnode` inside BaseModel's ```__init__```.

For Graph defined inside `BaseModel`, `BaseModel` will automatically call
the `_train_fprop` and `_test_fprop` within each layer inside its model.

```python
class MyLayer(BaseModel):
    def __init__(self):
        '''
         place all your layers inside here and define self.startnode and
         self.endnode
         example:
           layers = []
           layers.append(DenseBlock())
           layers.append(Flatten())
           layers.append(Linear())
           layers.append(Softmax())
           self.startnode = tg.StartNode(input_vars=[None])
           hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
           self.endnode = tg.EndNode(prev=[hn])
        '''
        pass
```
It is possible for BaseModel to return multiple outputs, example
```python
class MyLayerFork(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self):
       # a Y shape model, where we have one input and two outputs
       self.startnode = tg.StartNode(input_vars=[None])
       # first fork output
       layers = []
       layers.append(Linear())
       layers.append(Softmax())
       hn = tg.HiddenNode(prev=[self.startnode], layers=layers)

       # second fork output
       layers2 = []
       layers2.append(Linear())
       layers2.append(Softmax())
       hn2 = tg.HiddenNode(prev=[self.startnode], layers=layers2)

       # two forks outputs
       self.endnode = tg.EndNode(prev=[hn, h2])
```
In this case, a call to `train_fprop` will return two outputs
```python
mylayer = MylayerFork()
y1, y2 = mylayer.train_fprop(X_ph)
```

#### Customize inputs and outputs for BaseModel

Another way to customize your own inputs and outputs is to redefine `_train_fprop`
and `_test_fprop` within `BaseModel`.

The default `_train_fprop` and `_test_fprop` in `BaseModel` looks like this

```python
class BaseModel(Template):

    @staticmethod
    def check_y(y):
        if len(y) == 1:
            return y[0]
        elif len(y) > 1:
            return y
        else:
            raise Exception('{} is empty or not a list'.format(y))


    def _train_fprop(self, *state_belows):
        self.startnode.input_vars = state_belows
        graph = Graph(start=[self.startnode], end=[self.endnode])
        y = graph.train_fprop()
        return BaseModel.check_y(y)


    def _test_fprop(self, *state_belows):
        self.startnode.input_vars = state_belows
        graph = Graph(start=[self.startnode], end=[self.endnode])
        y = graph.test_fprop()
        return BaseModel.check_y(y)
```

for the `MyLayerFork` Model, for two inputs and two outputs, we can redefine it
with multiple `StartNodes` and `EndNodes` within `_train_fprop` and `_test_fprop`.

```python
class MyLayerFork(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self):
       # multiple inputs and multiple outputs

       self.startnode1 = tg.StartNode(input_vars=[None])
       self.startnode2 = tg.StartNode(input_vars=[None])

       layers1 = []
       layers1.append(Linear())
       layers1.append(Softmax())
       hn1 = tg.HiddenNode(prev=[self.startnode1], layers=layers)

       layers2 = []
       layers2.append(Linear())
       layers2.append(Softmax())
       hn2 = tg.HiddenNode(prev=[self.startnode2], layers=layers2)

       # two forks outputs
       self.endnode1 = tg.EndNode(prev=[hn1])
       self.endnode2 = tg.EndNode(prev=[hn2])


     def _train_fprop(self, input1, input2):
         self.startnode1.input_vars = [input1]
         self.startnode2.input_vars = [input2]
         graph = Graph(start=[self.startnode1, self.startnode2], end=[self.endnode1, self.endnode2])
         y = graph.train_fprop()
         return BaseModel.check_y(y)


     def _test_fprop(self, input1, input2):
         self.startnode1.input_vars = [input1]
         self.startnode2.input_vars = [input2]
         graph = Graph(start=[self.startnode1, self.startnode2], end=[self.endnode1, self.endnode2])
         y = graph.test_fprop()
         return BaseModel.check_y(y)


if __name__ == '__main__':
    model = MyLayerFork()
    y1, y2 = model.train_fprop(X1, X2)
```


### Merge
When we have more than one outputs from previous layer and we want to merge them,
we can use the `Merge` layer in [tensorgraph.layers.merge.Merge](./tensorgraph/layers/merge.py)
to merge multiple inputs into one.

```python
class Concat(Merge):
    @Merge.init_name_scope
    def __init__(self, axis=1):
        '''
        Concat which is a Merge layer is used to concat the list of states from
        layer below into one state
        '''
        self.axis = axis

    def _train_fprop(self, state_list):
        return tf.concat(axis=self.axis, values=state_list)
```

We can use `Merge` layer in conjunction with `BaseModel` layer with multiple outputs,
example

```python
class MyLayerMergeFork(BaseModel):
    def __init__(self):
        layers = []
        # fork layer from above example
        layers.append(MyLayerFork())
        # merge layer
        layers.append(Concat())
        self.startnode = tg.StartNode(input_vars=[None])
        hn = tg.HiddenNode(prev=[self.startnode], input_merge_mode=NoChange(), layers=layers)
        self.endnode = tg.EndNode(prev=[hn])
```

-----
## How TensorGraph Works?
In TensorGraph models, layers are put into nodes and nodes are connected together
into graph. When we create nodes and layers, we also initializes all the tensorflow
`Variables`, then we connect the nodes together to form a computational graph.
The initialization of `Variables` and the linking of `Variables` into a computational
graph are two separate steps. By splitting them into two separate steps, we ensure
the flexibility of building our computational graph without the worry of accidental
reinitialization of the `Variables`.
We defined three types of nodes

1. StartNode : for inputs to the graph
2. HiddenNode : for putting sequential layers inside
3. EndNode : for getting outputs from the model

We put all the sequential layers into a `HiddenNode`, `HiddenNode` can be connected
to another `HiddenNode` or `StartNode`, the nodes are connected together to form
an architecture. The graph always starts with `StartNode` and ends with `EndNode`.
Once we have defined an architecture, we can use the `Graph` object to connect the
path we want in the architecture, there can be multiple StartNodes (s1, s2, etc)
and multiple EndNodes (e1, e2, etc), we can define which path we want in the
entire architecture, example to link from `s2` to `e1`. The `StartNode` is where you place
your starting point, it can be a `placeholder`, a symbolic output from another graph,
or data output from `tfrecords`. `EndNode` is where you want to get an output from
the graph, where the output can be used to calculate loss or simply just a peek at the
outputs at that particular layer. Below shows an
[example](examples/example.py) of building a tensor graph.

-----
## Graph Example
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
                layers=[Linear(y2_dim), RELU()])
h2 = HiddenNode(prev=[s2],
                layers=[Linear(y2_dim), RELU()])
h3 = HiddenNode(prev=[h1, h2],
                input_merge_mode=Sum(),
                layers=[Linear(y1_dim), RELU()])
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
Finally build the graph by putting `StartNodes` and `EndNodes` into `Graph`, we
can choose to use the entire architecture by using all the `StartNodes` and `EndNodes`
and run the forward propagation to get symbolic output from train mode. The number
of outputs from `graph.train_fprop` is the same as the number of `EndNodes` put
into `Graph`
```python
graph = Graph(start=[s1, s2], end=[e1, e2])
o1, o2 = graph.train_fprop()
```
or we can choose which node to start and which node to end, example
```python
graph = Graph(start=[s2], end=[e1])
o1, = graph.train_fprop()
```
Finally build an optimizer to optimize the objective function
```python
o1_mse = tf.reduce_mean((y1 - o1)**2)
o2_mse = tf.reduce_mean((y2 - o2)**2)
mse = o1_mse + o2_mse
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
```

-----
## TensorGraph on Multiple GPUS
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
## Hierachical Softmax Example
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

h1 = HiddenNode(prev=[start], layers=[Linear(component_dim), Softmax()])
h2 = HiddenNode(prev=[h1], layers=[Linear(component_dim), Softmax()])
h3 = HiddenNode(prev=[h2], layers=[Linear(component_dim), Softmax()])
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
## Transfer Learning Example
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

h1 = HiddenNode(prev=[s1], layers=[Linear(shared_dim), RELU()])
h2 = HiddenNode(prev=[s2], layers=[Linear(shared_dim), RELU()])
h3 = HiddenNode(prev=[h1,h2], input_merge_mode=Sum(),
                layers=[Linear(y_dim), Softmax()])
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
