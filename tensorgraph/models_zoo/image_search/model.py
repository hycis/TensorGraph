"""
Purpose: model image search model
"""
import tensorflow as tf
import numpy as np
import os

from ...node import StartNode, HiddenNode, EndNode
from ...layers import BaseModel, Conv3D, RELU, Flatten, Linear, MaxPooling3D, Concat, BatchNormalization, Dropout

class Convbnrelu(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self,nfilters,kernel_size,stride):
        """
        define a model object.
        """
        layers = []
        layers.append(Conv3D(num_filters=nfilters, kernel_size=kernel_size, stride=stride, padding='SAME', stddev=0.1))
        layers.append(BatchNormalization())
        layers.append(RELU())

        self.startnode = StartNode(input_vars=[None])
        model_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[model_hn])

class Inception(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, ks1, ks2, channel_1, channel_2_1x1, channel_2_3x3, channel_3_1x1, channel_3_5x5, poolsize_4_max, channel_4_1x1, stride_4_max):
        """
        args:
            inp: input layer
            ks1:  3x3 conv strides
            ks2: 5x5 conv strides
        return:
            incept: output of inception layer
        """
        #print('outputSize = ', channel_1+channel_2_3x3+channel_3_5x5+channel_4_1x1)

        self.startnode = StartNode(input_vars=[None])
        def incept_layers(nfilters,kernel_size,stride):
            layers = []
            layers.append(Conv3D(num_filters=nfilters, kernel_size=kernel_size, stride=stride, padding='SAME', stddev=0.1))
            layers.append(BatchNormalization())
            layers.append(RELU())
            return layers

        incept = []
        if channel_1>0:
            conv1 = HiddenNode(prev=[self.startnode], layers=incept_layers(channel_1, (1, 1, 1), (1, 1, 1)))
        # else:
        #     conv1 = StartNode(input_vars=[None])

        conv3a = HiddenNode(prev=[self.startnode], layers=incept_layers(channel_2_1x1, (1, 1, 1), (1, 1, 1)))
        conv3 = HiddenNode(prev=[conv3a], layers=incept_layers(channel_2_3x3, (3, 3, 3), ks1))

        conv5a = HiddenNode(prev=[self.startnode], layers=incept_layers(channel_3_1x1, (1, 1, 1), (1, 1, 1)))
        conv5 = HiddenNode(prev=[conv5a], layers=incept_layers(channel_3_5x5, (5, 5, 5), ks2))

        layer_pool=[]
        layer_pool.append(MaxPooling3D(poolsize_4_max, stride_4_max, 'SAME'))

        pool = HiddenNode(prev=[self.startnode], layers=layer_pool)
        pool_conv = HiddenNode(prev=[pool], layers=incept_layers(channel_4_1x1, (1, 1, 1), (1, 1, 1)))

        if channel_1>0:
            incept = EndNode(prev=[conv1,conv3,conv5,pool_conv], input_merge_mode=Concat(axis=-1))
        else:
            incept = EndNode(prev=[conv3,conv5,pool_conv], input_merge_mode=Concat(axis=-1))

        self.endnode = incept

class Image_Search_Model(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, keep_probability=0.5, bottleneck_layer_size=128):
        """ Define an inference network based on inception modules
        args:
            keep_probability: probability of dropout
            bottleneck_layer_size: dimension of output embeddings
        """
        net = []
        net.append(Convbnrelu(64, (7, 7, 7), (1, 2, 2)))
        net.append(MaxPooling3D((3, 3, 3), (1, 2, 2), 'SAME'))
        net.append(Convbnrelu(64, (1, 1, 1), (1, 1, 1)))
        net.append(Convbnrelu(192, (3, 3, 3), (1, 1, 1)))
        net.append(MaxPooling3D((3, 3, 3), (1, 2, 2), 'SAME'))

        net.append(Inception((1,1,1), (1,1,1), 64, 96, 128, 16, 32, (3,3,3), 32, (1,1,1)))
        net.append(Dropout(keep_probability))
        net.append(Inception((2,2,2), (2,2,2), 0, 128, 256, 32, 64, (3,3,3), 256, (2,2,2)))
        net.append(Dropout(keep_probability))
        net.append(Inception((1,1,1), (1,1,1), 256, 96, 192, 32, 64, (3,3,3), 128, (1,1,1)))
        net.append(Dropout(keep_probability))
        net.append(Inception((2,2,2), (2,2,2), 0, 80, 128, 32, 64, (3,3,3), 320, (2,2,2)))
        net.append(Dropout(keep_probability))
        net.append(Inception((1,1,1), (1,1,1), 384, 192, 384, 48, 128, (3,3,3), 128, (1,1,1)))
        net.append(MaxPooling3D((5, 5, 5), (1, 1, 1), 'VALID'))
        #net.append(MaxPooling3D((1, 1, 1), (1, 1, 1), 'SAME'))

        net.append(Flatten())
        net.append(Linear(this_dim=bottleneck_layer_size))
        net.append(Dropout(keep_probability))

        self.startnode = StartNode(input_vars=[None])
        h = HiddenNode(prev=[self.startnode], layers=net)
        self.endnode = EndNode(prev=[h])
