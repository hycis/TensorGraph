import tensorflow as tf
import numpy as np

from ...graph import Graph
from ...node import StartNode, HiddenNode, EndNode
from ...layers import BaseModel, Conv3D, RELU, Flatten, Linear, MaxPooling3D, Concat, BatchNormalization

class AIRNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self):
        '''
        The model takes in two input tensors. The lower layers consists of
        several convolutional blocks with both inputs passing through the
        the same pathway (shared weights). The outputs of the encoder are
        flatten and concatenated before passed into the fully connected layers.

        Args:
            d: Depth of input tensors.
            h: Height of input tensors.
            w: Width of input tensors.
            c: Number of channels of input tensors.
        '''

        # Encoder
        shared_layers = []

        # 2D convolution
        shared_layers.append(Conv3D(num_filters=8, kernel_size=(1,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='VALID'))

        shared_layers.append(Conv3D(num_filters=16, kernel_size=(1,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='VALID'))

        shared_layers.append(Conv3D(num_filters=32, kernel_size=(1,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='VALID'))

        shared_layers.append(Conv3D(num_filters=64, kernel_size=(1,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(1,2,2), stride=(1,2,2), padding='VALID'))

        # 3D convolution
        shared_layers.append(Conv3D(num_filters=128, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(2,2,2), stride=(2,2,2), padding='VALID'))

        shared_layers.append(Conv3D(num_filters=256, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(2,2,2), stride=(2,2,2), padding='VALID'))

        shared_layers.append(Conv3D(num_filters=512, kernel_size=(3,3,3), stride=(1,1,1), padding='SAME'))
        shared_layers.append(BatchNormalization())
        shared_layers.append(RELU())
        shared_layers.append(MaxPooling3D(poolsize=(2,2,2), stride=(2,2,2), padding='VALID'))

        shared_layers.append(Flatten())

        # Fully connected
        layers = []
        layers.append(Linear(this_dim=1024))
        layers.append(BatchNormalization())
        layers.append(Linear(this_dim=512))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Linear(this_dim=128))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Linear(this_dim=64))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Linear(this_dim=12, b=tf.Variable(initial_value=np.array([[1.,0,0,0],[0,1.,0,0],[0,0,1.,0]]).astype('float32').flatten())))

        self.startnode1 = StartNode(input_vars=[None])
        self.startnode2 = StartNode(input_vars=[None])

        h1 = HiddenNode(prev=[self.startnode1], layers=shared_layers)
        h2 = HiddenNode(prev=[self.startnode2], layers=shared_layers)
        h3 = HiddenNode(prev=[h1,h2], input_merge_mode=Concat(), layers=layers)

        self.endnode = EndNode(prev=[h3])

    def _train_fprop(self, state_below1, state_below2):
        '''
        Forward propogation through the layer during training.

        Args:
            state_below1: Tensor of shape [batch, d, h, w, c].
            state_below2: Same shape as state_below1.

        Returns:
            Tensor of shape [batch, 12].
        '''
        self.startnode1.input_vars = [state_below1]
        self.startnode2.input_vars = [state_below2]
        graph = Graph(start=[self.startnode1, self.startnode2], end=[self.endnode])
        y = graph.train_fprop()
        return BaseModel.check_y(y)

    def _test_fprop(self, state_below1, state_below2):
        '''
        Forward propogation through the layer during testing.

        Args:
            state_below1: Tensor of shape [batch, d, h, w, c].
            state_below2: Same shape as state_below1.

        Returns:
            Tensor of shape [batch, 12].
        '''
        self.startnode1.input_vars = [state_below1]
        self.startnode2.input_vars = [state_below2]
        graph = Graph(start=[self.startnode1, self.startnode2], end=[self.endnode])
        y = graph.test_fprop()
        return BaseModel.check_y(y)
