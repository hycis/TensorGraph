
from ..sequential import Sequential
from ..graph import Graph
from ..node import StartNode, HiddenNode, EndNode
from . import Conv2D, MaxPooling, RELU, ELU, BatchNormalization, Template, Sum, \
              Concat, AvgPooling, Conv2D_Transpose, Dropout
from ..utils import same_nd, valid_nd, devalid_nd, desame_nd
import numpy as np


class BaseModel(Template):

    @staticmethod
    def check_y(y):
        if len(y) == 1:
            return y[0]
        elif len(y) > 1:
            return y
        else:
            raise Exception('{} is empty or not a list'.format(y))


    def _train_fprop(self, state_below):
        self.startnode.input_vars = [state_below]
        graph = Graph(start=[self.startnode], end=[self.endnode])
        y = graph.train_fprop()
        return BaseModel.check_y(y)


    def _test_fprop(self, state_below):
        self.startnode.input_vars = [state_below]
        graph = Graph(start=[self.startnode], end=[self.endnode])
        y = graph.test_fprop()
        return BaseModel.check_y(y)


class VGG16(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape):
        '''
        REFERENCE: Very Deep Convolutional Networks for Large-Scale Image Recognition
                   (https://arxiv.org/abs/1409.1556)
        '''
        layers = []
        # block 1
        layers.append(Conv2D(input_channels, num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(input_shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[64]))
        layers.append(RELU())
        layers.append(Conv2D(64, num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[64]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 2
        layers.append(Conv2D(64, num_filters=128, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[128]))
        layers.append(RELU())
        layers.append(Conv2D(128, num_filters=128, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[128]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 3
        layers.append(Conv2D(128, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(Conv2D(256, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(Conv2D(256, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 4
        layers.append(Conv2D(256, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 5
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        assert np.prod(shape) > 0, 'output shape {} is <= 0'.format(shape)
        self.output_shape = shape
        self.output_channels = 512

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class VGG19(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape):
        '''
        REFERENCE: Very Deep Convolutional Networks for Large-Scale Image Recognition
                   (https://arxiv.org/abs/1409.1556)
        '''
        layers = []
        # block 1
        layers.append(Conv2D(input_channels, num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(input_shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[64]))
        layers.append(RELU())
        layers.append(Conv2D(64, num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[64]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 2
        layers.append(Conv2D(64, num_filters=128, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[128]))
        layers.append(RELU())
        layers.append(Conv2D(128, num_filters=128, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[128]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 3
        layers.append(Conv2D(128, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(Conv2D(256, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(Conv2D(256, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(Conv2D(256, num_filters=256, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[256]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 4
        layers.append(Conv2D(256, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        # block 5
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(Conv2D(512, num_filters=512, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[512]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))

        assert np.prod(shape) > 0, 'output shape {} is <= 0'.format(shape)
        self.output_shape = shape
        self.output_channels = 512

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class ResNetBase(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape, config):
        '''
        REFERENCE: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        PARAMS:
            config (list of ints): a list of 4 number of layers for each identity block
        '''

        layers = []
        layers.append(Conv2D(input_channels, num_filters=64, kernel_size=(7,7), stride=(2,2), padding='SAME'))
        shape = same_nd(input_shape, kernel_size=(7,7), stride=(2,2))
        layers.append(BatchNormalization(input_shape=shape+[64]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(3,3), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(3,3), stride=(2,2))

        shortcut = ShortCutBlock(input_channels=64, filters=[64, 64, 256],
                                 input_shape=shape, kernel_size=(3,3), stride=(1,1))
        layers.append(shortcut)
        identity = IdentityBlock(input_channels=256, filters=[64, 64],
                                 input_shape=shortcut.output_shape, nlayers=config[0])
        layers.append(identity)

        shortcut = ShortCutBlock(input_channels=256, filters=[128, 128, 512],
                                 input_shape=identity.output_shape, kernel_size=(3,3), stride=(2,2))
        layers.append(shortcut)
        identity = IdentityBlock(input_channels=512, filters=[128, 128],
                                 input_shape=shortcut.output_shape, nlayers=config[1])
        layers.append(identity)

        shortcut = ShortCutBlock(input_channels=512, filters=[256, 256, 1024],
                                 input_shape=identity.output_shape, kernel_size=(3,3), stride=(2,2))
        layers.append(shortcut)
        identity = IdentityBlock(input_channels=1024, filters=[256, 256],
                                 input_shape=shortcut.output_shape, nlayers=config[2])
        layers.append(identity)

        shortcut = ShortCutBlock(input_channels=1024, filters=[512, 512, 2048],
                                 input_shape=identity.output_shape, kernel_size=(3,3), stride=(2,2))
        layers.append(shortcut)
        identity = IdentityBlock(input_channels=2048, filters=[512, 512],
                                 input_shape=shortcut.output_shape, nlayers=config[3])
        layers.append(identity)

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])
        assert np.prod(shape) > 0, 'output shape {} is <= 0'.format(shape)
        self.output_shape = identity.output_shape
        self.output_channels = 2048


class ResNetSmall(ResNetBase):

    @ResNetBase.init_name_scope
    def __init__(self, input_channels, input_shape, config):
        '''
        REFERENCE: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        PARAMS:
            config (list of ints): a list of 2 number of layers for each identity block
        '''
        layers = []
        layers.append(Conv2D(input_channels, num_filters=64, kernel_size=(7,7), stride=(2,2), padding='SAME'))
        shape = same_nd(input_shape, kernel_size=(7,7), stride=(2,2))
        layers.append(BatchNormalization(input_shape=shape+[64]))
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(3,3), stride=(2,2), padding='VALID'))
        shape = valid_nd(shape, kernel_size=(3,3), stride=(2,2))

        shortcut = ShortCutBlock(input_channels=64, filters=[64, 64, 128],
                                 input_shape=shape, kernel_size=(3,3), stride=(1,1))
        layers.append(shortcut)
        identity = IdentityBlock(input_channels=128, filters=[64, 64],
                                 input_shape=shortcut.output_shape, nlayers=config[0])
        layers.append(identity)

        shortcut = ShortCutBlock(input_channels=128, filters=[128, 128, 128],
                                 input_shape=identity.output_shape, kernel_size=(3,3), stride=(2,2))
        layers.append(shortcut)
        identity = IdentityBlock(input_channels=128, filters=[128, 128],
                                 input_shape=shortcut.output_shape, nlayers=config[1])
        layers.append(identity)

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])
        assert np.prod(shape) > 0, 'output shape {} is <= 0'.format(shape)
        self.output_shape = identity.output_shape
        self.output_channels = 128


class ResNet50(ResNetBase):
    def __init__(self, input_channels, input_shape):
        super(ResNet50, self).__init__(input_channels, input_shape, config=[2,3,5,2])


class ResNet101(ResNetBase):
    def __init__(self, input_channels, input_shape):
        super(ResNet101, self).__init__(input_channels, input_shape, config=[2,3,22,2])


class ResNet152(ResNetBase):
    def __init__(self, input_channels, input_shape):
        super(ResNet101, self).__init__(input_channels, input_shape, config=[2,7,35,2])


class ShortCutBlock(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape, filters, kernel_size, stride):
        '''
        DESCRIPTION:
            The shortcut block in Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        PARAMS:
            config (list of ints): a list of 4 number of layers for each identity block
            filters (list of 3 ints): number of filters in different CNN layers
            kernel_size (tuple of 2 ints)
            stride (tuple of 2 ints)
        '''
        assert isinstance(filters, (list, tuple)) and len(filters) == 3
        f1, f2, f3 = filters
        layers = []
        layers.append(Conv2D(input_channels, num_filters=f1, kernel_size=(1,1), stride=stride, padding='VALID'))
        shape = valid_nd(input_shape, kernel_size=(1,1), stride=stride)
        layers.append(BatchNormalization(input_shape=shape+[f1]))
        layers.append(RELU())

        layers.append(Conv2D(f1, num_filters=f2, kernel_size=kernel_size, stride=(1,1), padding='SAME'))
        shape = same_nd(shape, kernel_size=kernel_size, stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[f2]))
        layers.append(RELU())

        layers.append(Conv2D(f2, num_filters=f3, kernel_size=(1,1), stride=(1,1), padding='VALID'))
        shape = same_nd(shape, kernel_size=(1,1), stride=(1,1))
        layers.append(BatchNormalization(input_shape=shape+[f3]))
        layers.append(RELU())

        shortcuts = []
        shortcuts.append(Conv2D(input_channels, num_filters=f3, kernel_size=(1,1), stride=stride, padding='VALID'))
        shape = valid_nd(input_shape, kernel_size=(1,1), stride=stride)
        shortcuts.append(BatchNormalization(input_shape=shape+[f3]))
        shortcuts.append(RELU())

        self.startnode = StartNode(input_vars=[None])
        conv_hn = HiddenNode(prev=[self.startnode], layers=layers)
        shortcuts_hn = HiddenNode(prev=[self.startnode], layers=shortcuts)
        out_hn = HiddenNode(prev=[conv_hn, shortcuts_hn], input_merge_mode=Sum())
        self.endnode = EndNode(prev=[out_hn])
        self.output_shape = shape
        self.output_channels = f3


class IdentityBlock(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape, nlayers=2, filters=[32, 64]):
        '''
        DESCRIPTION:
            one identity block of a resnet in the paper Deep Residual Learning
            for Image Recognition (https://arxiv.org/abs/1512.03385)
        PARAMS:
            nlayers (int): number recurrent cycles within one identity block
            filters (list of 2 ints): number of filters within one identity block
        '''
        assert isinstance(filters, (list, tuple)) and len(filters) == 2
        f1, f2 = filters
        def identity_layer(in_hn, shape):
            layers = []
            layers.append(Conv2D(input_channels, num_filters=f1, kernel_size=(1,1), stride=(1,1), padding='VALID'))
            shape = valid_nd(shape, kernel_size=(1,1), stride=(1,1))
            layers.append(BatchNormalization(input_shape=shape+[f1]))
            layers.append(RELU())

            layers.append(Conv2D(f1, num_filters=f2, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
            layers.append(BatchNormalization(input_shape=shape+[f2]))
            layers.append(RELU())

            layers.append(Conv2D(f2, num_filters=input_channels, kernel_size=(1,1), stride=(1,1), padding='VALID'))
            shape = same_nd(shape, kernel_size=(1,1), stride=(1,1))
            layers.append(BatchNormalization(input_shape=shape+[input_channels]))
            layers.append(RELU())
            out_hn = HiddenNode(prev=[in_hn], layers=layers)
            return out_hn, shape

        self.startnode = in_hn = StartNode(input_vars=[None])

        shape = input_shape
        for _ in range(nlayers):
            out_hn, shape = identity_layer(in_hn, shape)
            in_hn = HiddenNode(prev=[out_hn, in_hn], input_merge_mode=Sum())

        self.endnode = EndNode(prev=[in_hn])
        self.output_shape = shape
        self.output_channels = input_channels


class DenseBlock(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape, growth_rate, nlayers):
        '''
        DESCRIPTION:
            one dense block from the densely connected CNN (Densely Connected
            Convolutional Networks https://arxiv.org/abs/1608.06993)
        PARAMS:
            growth_rate (int): number of filters to grow inside one denseblock
            nlayers (int): number of layers in one block, one layer refers to
                one group of batchnorm, relu and conv2d
        '''

        def _conv_layer(in_hn, shape, in_channel):
            layers = []
            layers.append(Conv2D(in_channel, num_filters=growth_rate, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            layers.append(BatchNormalization(input_shape=shape+[growth_rate]))
            layers.append(RELU())
            shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
            out_hn = HiddenNode(prev=[in_hn], layers=layers)
            out_hn = HiddenNode(prev=[in_hn, out_hn],
                                   input_merge_mode=Concat(axis=-1))
            return out_hn, shape, growth_rate+in_channel

        self.startnode = in_hn = StartNode(input_vars=[None])
        shape = input_shape
        in_channel = input_channels
        for _ in range(nlayers):
            in_hn, shape, in_channel = _conv_layer(in_hn, shape, in_channel)
        self.endnode = EndNode(prev=[in_hn])
        self.output_channels = in_channel
        self.output_shape = shape


class TransitionLayer(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape):
        '''
        DESCRIPTION:
            The transition layer of densenet (Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993)
        '''
        layers = []
        layers.append(Conv2D(input_channels, num_filters=input_channels, kernel_size=(1,1), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization(input_shape=input_shape+[input_channels]))
        layers.append(RELU())
        layers.append(AvgPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(input_shape, kernel_size=(2,2), stride=(2,2))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])
        self.output_channels = input_channels
        self.output_shape = shape


class DenseNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape, ndense=3, growth_rate=12, nlayer1blk=12):
        '''
        REFERENCE: Densely Connected Convolutional Networks (https://arxiv.org/abs/1608.06993)
        PARAMS:
            ndense (int): number of dense blocks
            nlayer1blk (int): number of layers in one block, one layer refers to
                one group of batchnorm, relu and conv2d
        '''
        layers = []
        layers.append(Conv2D(input_channels, num_filters=16, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        shape = same_nd(input_shape, kernel_size=(3,3), stride=(1,1))
        layers.append(BatchNormalization(input_shape=list(shape)+[16]))
        layers.append(RELU())

        dense = DenseBlock(input_channels=16, input_shape=shape, growth_rate=growth_rate, nlayers=nlayer1blk)
        layers.append(dense)
        transit = TransitionLayer(input_channels=dense.output_channels, input_shape=dense.output_shape)
        layers.append(transit)

        for _ in range(ndense-1):
            dense = DenseBlock(transit.output_channels, transit.output_shape, growth_rate, nlayer1blk)
            layers.append(dense)
            transit = TransitionLayer(dense.output_channels, dense.output_shape)
            layers.append(transit)

        dense = DenseBlock(transit.output_channels, transit.output_shape, growth_rate, nlayer1blk)
        layers.append(dense)
        # layers.append(AvgPooling(poolsize=dense.output_shape, stride=(1,1), padding='VALID'))

        assert np.prod(dense.output_shape) > 0, 'output shape {} is <= 0'.format(dense.output_shape)
        self.startnode = StartNode(input_vars=[None])
        model_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[model_hn])
        self.output_shape = dense.output_shape
        self.output_channels = dense.output_channels


# TODO
# class FeaturePyramidNetwork(Template):
# '''
# reference: Feature Pyramid Networks for Object Detection (https://arxiv.org/abs/1612.03144)
# '''
#     pass

# TODO
# class PyramidPoolingModule(Template):
# '''reference: Pyramid Scene Parsing Network (https://arxiv.org/abs/1612.01105)
# '''
#     pass


class UNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_channels, input_shape):

        def _encode_block(in_hn, shape, in_ch, out_ch):
            blk = []
            blk.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
            shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))
            blk.append(Conv2D(input_channels=in_ch, num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
            blk.append(BatchNormalization(input_shape=shape+[out_ch]))
            blk.append(Conv2D(input_channels=out_ch, num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
            blk.append(BatchNormalization(input_shape=shape+[out_ch]))
            out_hn = HiddenNode(prev=[in_hn], layers=blk)
            return out_hn, shape


        def _merge_decode_block(deblk_hn, blk_hn, input_shape, target_shape, in_ch, out_ch):
            blk = []
            blk.append(Conv2D(input_channels=in_ch, num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization(input_shape=input_shape+[out_ch]))
            blk.append(Conv2D(input_channels=out_ch, num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization(input_shape=input_shape+[out_ch]))
            blk.append(Conv2D_Transpose(input_channels=out_ch, num_filters=out_ch, output_shape=target_shape,
                                                kernel_size=(2,2), stride=(2,2), padding='SAME'))
            dshape = desame_nd(input_shape, kernel_size=(2,2), stride=(2,2))
            blk.append(ELU())
            blk.append(BatchNormalization(input_shape=input_shape+[out_ch]))
            out_hn = HiddenNode(prev=[deblk_hn, blk_hn],
                                   input_merge_mode=Concat(axis=-1),
                                   layers=blk)
            return out_hn


        # encoding
        blk1 = []
        blk1.append(Conv2D(input_channels, num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk1.append(ELU())
        shape = same_nd(input_shape, kernel_size=(3,3), stride=(1,1))
        blk1.append(BatchNormalization(input_shape=shape+[64]))
        blk1.append(Conv2D(input_channels=64, num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk1.append(ELU())
        b1_shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        blk1.append(BatchNormalization(input_shape=b1_shape+[64]))

        self.startnode = StartNode(input_vars=[None])
        blk1_hn = HiddenNode(prev=[self.startnode], layers=blk1)
        blk2_hn, b2_shape = _encode_block(blk1_hn, b1_shape, 64, 128)
        blk3_hn, b3_shape = _encode_block(blk2_hn, b2_shape, 128, 256)
        blk4_hn, b4_shape = _encode_block(blk3_hn, b3_shape, 256, 512)

        # downsampling + conv
        deblk4 = []
        deblk4.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(b4_shape, kernel_size=(2,2), stride=(2,2))
        deblk4.append(Conv2D(input_channels=512, num_filters=1024, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        deblk4.append(ELU())
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        deblk4.append(BatchNormalization(input_shape=shape+[1024]))
        deblk4.append(Conv2D(input_channels=1024, num_filters=1024, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        deblk4.append(ELU())
        out_shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        deblk4.append(BatchNormalization(input_shape=out_shape+[1024]))

        # deconvolve
        deblk4.append(Conv2D_Transpose(input_channels=1024, num_filters=1024, output_shape=b4_shape,
                                       kernel_size=(2,2), stride=(2,2), padding='SAME'))
        deblk4.append(ELU())
        deblk4.append(BatchNormalization(input_shape=shape+[1024]))

        # decode and merge
        deblk4_hn = HiddenNode(prev=[blk4_hn], layers=deblk4)
        deblk3_hn = _merge_decode_block(deblk4_hn, blk4_hn, b4_shape, b3_shape, in_ch=1024+512, out_ch=256)
        deblk2_hn = _merge_decode_block(deblk3_hn, blk3_hn, b3_shape, b2_shape, in_ch=256+256, out_ch=128)
        deblk1_hn = _merge_decode_block(deblk2_hn, blk2_hn, b2_shape, b1_shape, in_ch=128+128, out_ch=64)

        # reduce channels
        blk = []
        blk.append(Conv2D(input_channels=64+64, num_filters=32, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk.append(ELU())
        blk.append(BatchNormalization(input_shape=list(input_shape)+[32]))
        blk.append(Conv2D(input_channels=32, num_filters=16, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk.append(ELU())
        blk.append(BatchNormalization(input_shape=list(input_shape)+[16]))
        deblk0_hn = HiddenNode(prev=[deblk1_hn, blk1_hn],
                                  input_merge_mode=Concat(axis=-1),
                                  layers=blk)

        self.endnode = EndNode(prev=[deblk0_hn])
        self.output_shape = input_shape
        self.output_channels = 16
        assert np.prod(shape) > 0, 'output shape {} is <= 0'.format(shape)
