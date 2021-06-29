
from ..sequential import Sequential
from ..graph import Graph
from ..node import StartNode, HiddenNode, EndNode
from . import Conv2D, MaxPooling, RELU, ELU, BatchNormalization, Sum, \
              Concat, AvgPooling, Conv2D_Transpose, Dropout, BaseModel
from ..utils import same_nd, valid_nd, devalid_nd, desame_nd
import numpy as np


class VGG16(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self):
        '''
        Reference:
            Very Deep Convolutional Networks for Large-Scale Image Recognition
            (https://arxiv.org/abs/1409.1556)
        '''
        filters = [64, 128, 256, 512]
        layers = []
        # block 1
        layers.append(Conv2D(num_filters=filters[0], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[0], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 2
        layers.append(Conv2D(num_filters=filters[1], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[1], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 3
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 4
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 5
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class VGG19(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self):
        '''
        Reference:
            Very Deep Convolutional Networks for Large-Scale Image Recognition
            (https://arxiv.org/abs/1409.1556)
        '''
        filters = [64, 128, 256, 512]
        layers = []
        # block 1
        layers.append(Conv2D(num_filters=filters[0], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[0], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 2
        layers.append(Conv2D(num_filters=filters[1], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[1], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 3
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[2], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 4
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        # block 5
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(Conv2D(num_filters=filters[3], kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class ResNetBase(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, config):
        '''
        Reference:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

        Args:
            config (list of ints): a list of 4 number of layers for each identity block
        '''

        layers = []
        layers.append(Conv2D(num_filters=64, kernel_size=(7,7), stride=(2,2), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(3,3), stride=(2,2), padding='VALID'))

        layers.append(ShortCutBlock(filters=[64, 64, 256], kernel_size=(3,3), stride=(1,1)))
        layers.append(IdentityBlock(filters=[64, 64], nlayers=config[0]))

        layers.append(ShortCutBlock(filters=[128, 128, 512], kernel_size=(3,3), stride=(2,2)))
        layers.append(IdentityBlock(filters=[128, 128], nlayers=config[1]))

        layers.append(ShortCutBlock(filters=[256, 256, 1024], kernel_size=(3,3), stride=(2,2)))
        layers.append(IdentityBlock(filters=[256, 256], nlayers=config[2]))

        layers.append(ShortCutBlock(filters=[512, 512, 2048], kernel_size=(3,3), stride=(2,2)))
        layers.append(IdentityBlock(filters=[512, 512], nlayers=config[3]))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class ResNetSmall(ResNetBase):

    @ResNetBase.init_name_scope
    def __init__(self, config):
        '''
        Reference:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

        Args:
            config (list of ints): a list of 2 number of layers for each identity block
        '''
        layers = []
        layers.append(Conv2D(num_filters=64, kernel_size=(7,7), stride=(2,2), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(MaxPooling(poolsize=(3,3), stride=(2,2), padding='VALID'))

        layers.append(ShortCutBlock(filters=[64, 64, 128], kernel_size=(3,3), stride=(1,1)))
        layers.append(IdentityBlock(filters=[64, 64], nlayers=config[0]))

        layers.append(ShortCutBlock(filters=[128, 128, 128], kernel_size=(3,3), stride=(2,2)))
        layers.append(IdentityBlock(filters=[128, 128], nlayers=config[1]))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class ResNet50(ResNetBase):
    def __init__(self, config=[2,3,5,2]):
        super(ResNet50, self).__init__(config)


class ResNet101(ResNetBase):
    def __init__(self, config=[2,3,22,2]):
        super(ResNet101, self).__init__(config)


class ResNet152(ResNetBase):
    def __init__(self, config=[2,7,35,2]):
        super(ResNet101, self).__init__(config)


class ShortCutBlock(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, filters, kernel_size=(3,3), stride=(1,1)):
        '''
        Reference:
            The shortcut block in Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

        Args:
            filters (list of 3 ints): number of filters in different CNN layers
            kernel_size (tuple of 2 ints)
            stride (tuple of 2 ints)
        '''
        assert isinstance(filters, (list, tuple)) and len(filters) == 3
        f1, f2, f3 = filters
        layers = []
        layers.append(Conv2D(num_filters=f1, kernel_size=(1,1), stride=stride, padding='VALID'))
        layers.append(BatchNormalization())
        layers.append(RELU())

        layers.append(Conv2D(num_filters=f2, kernel_size=kernel_size, stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())

        layers.append(Conv2D(num_filters=f3, kernel_size=(1,1), stride=(1,1), padding='VALID'))
        layers.append(BatchNormalization())
        layers.append(RELU())

        shortcuts = []
        shortcuts.append(Conv2D(num_filters=f3, kernel_size=(1,1), stride=stride, padding='VALID'))
        shortcuts.append(BatchNormalization())
        shortcuts.append(RELU())

        self.startnode = StartNode(input_vars=[None])
        conv_hn = HiddenNode(prev=[self.startnode], layers=layers)
        shortcuts_hn = HiddenNode(prev=[self.startnode], layers=shortcuts)
        out_hn = HiddenNode(prev=[conv_hn, shortcuts_hn], input_merge_mode=Sum())
        self.endnode = EndNode(prev=[out_hn])
        self.output_channels = f3


class IdentityBlock(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nlayers=2, filters=[32, 64]):
        '''
        Description:
            one identity block of a resnet in the paper Deep Residual Learning
            for Image Recognition (https://arxiv.org/abs/1512.03385)

        Args:
            nlayers (int): number recurrent cycles within one identity block
            filters (list of 2 ints): number of filters within one identity block
        '''
        assert isinstance(filters, (list, tuple)) and len(filters) == 2
        self.filters = filters
        self.nlayers = nlayers


    @BaseModel.init_name_scope
    def __init_var__(self, state_below):
        b, h, w, c = state_below.shape
        c = int(c)
        f1, f2 = self.filters
        def identity_layer(in_hn):
            layers = []
            layers.append(Conv2D(num_filters=f1, kernel_size=(1,1), stride=(1,1), padding='VALID'))
            layers.append(BatchNormalization())
            layers.append(RELU())

            layers.append(Conv2D(num_filters=f2, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            layers.append(BatchNormalization())
            layers.append(RELU())

            layers.append(Conv2D(num_filters=c, kernel_size=(1,1), stride=(1,1), padding='VALID'))
            layers.append(BatchNormalization())
            layers.append(RELU())
            out_hn = HiddenNode(prev=[in_hn], layers=layers)
            return out_hn

        self.startnode = in_hn = StartNode(input_vars=[None])
        for _ in range(self.nlayers):
            out_hn = identity_layer(in_hn)
            in_hn = HiddenNode(prev=[out_hn, in_hn], input_merge_mode=Sum())
        self.endnode = EndNode(prev=[in_hn])



class DenseBlock(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, growth_rate, nlayers):
        '''
        Description:
            one dense block from the densely connected CNN (Densely Connected
            Convolutional Networks https://arxiv.org/abs/1608.06993)

        Args:
            growth_rate (int): number of filters to grow inside one denseblock
            nlayers (int): number of layers in one block, one layer refers to
                one group of batchnorm, relu and conv2d
        '''

        def _conv_layer(in_hn):
            layers = []
            layers.append(Conv2D(num_filters=growth_rate, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            layers.append(BatchNormalization())
            layers.append(RELU())
            out_hn = HiddenNode(prev=[in_hn], layers=layers)
            out_hn = HiddenNode(prev=[in_hn, out_hn],
                                   input_merge_mode=Concat(axis=-1))
            return out_hn
        self.startnode = in_hn = StartNode(input_vars=[None])
        for _ in range(nlayers):
            in_hn = _conv_layer(in_hn)
        self.endnode = EndNode(prev=[in_hn])


class TransitionLayer(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, num_filters):
        '''
        Description:
            The transition layer of densenet (Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993)
        '''
        layers = []
        layers.append(Conv2D(num_filters=num_filters, kernel_size=(1,1), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(AvgPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])


class DenseNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, ndense=3, growth_rate=12, nlayer1blk=12):
        '''
        Reference:
            Densely Connected Convolutional Networks (https://arxiv.org/abs/1608.06993)

        Args:
            ndense (int): number of dense blocks
            nlayer1blk (int): number of layers in one block, one layer refers to
                one group of batchnorm, relu and conv2d
        '''
        layers = []
        layers.append(Conv2D(num_filters=16, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        layers.append(DenseBlock(growth_rate=growth_rate, nlayers=nlayer1blk))
        layers.append(TransitionLayer(num_filters=16))

        for _ in range(ndense-1):
            layers.append(DenseBlock(growth_rate=growth_rate, nlayers=nlayer1blk))
            layers.append(TransitionLayer(num_filters=16))

        layers.append(DenseBlock(growth_rate=growth_rate, nlayers=nlayer1blk))
        # layers.append(AvgPooling(poolsize=dense.output_shape, stride=(1,1), padding='VALID'))

        self.startnode = StartNode(input_vars=[None])
        model_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[model_hn])



# TODO
# class FeaturePyramidNetwork(BaseModel):
# '''
# reference: Feature Pyramid Networks for Object Detection (https://arxiv.org/abs/1612.03144)
# '''
#     pass

# TODO
# class PyramidPoolingModule(BaseModel):
# '''reference: Pyramid Scene Parsing Network (https://arxiv.org/abs/1612.01105)
# '''
#     pass


class UNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_shape):

        def _encode_block(in_hn, shape, out_ch):
            blk = []
            blk.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
            shape = valid_nd(shape, kernel_size=(2,2), stride=(2,2))
            blk.append(Conv2D(num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
            blk.append(BatchNormalization())
            blk.append(Conv2D(num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
            blk.append(BatchNormalization())
            out_hn = HiddenNode(prev=[in_hn], layers=blk)
            return out_hn, shape


        def _merge_decode_block(deblk_hn, blk_hn, out_ch):
            blk = []
            blk.append(Conv2D(num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization())
            blk.append(Conv2D(num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization())
            blk.append(Conv2D_Transpose(num_filters=out_ch,
                                        kernel_size=(2,2), stride=(2,2), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization())
            out_hn = HiddenNode(prev=[deblk_hn, blk_hn],
                                   input_merge_mode=Concat(axis=-1),
                                   layers=blk)
            return out_hn


        # encoding
        blk1 = []
        blk1.append(Conv2D(num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk1.append(ELU())
        shape = same_nd(input_shape, kernel_size=(3,3), stride=(1,1))
        blk1.append(BatchNormalization())
        blk1.append(Conv2D(num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk1.append(ELU())
        b1_shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        blk1.append(BatchNormalization())

        self.startnode = StartNode(input_vars=[None])
        blk1_hn = HiddenNode(prev=[self.startnode], layers=blk1)
        blk2_hn, b2_shape = _encode_block(blk1_hn, b1_shape, 128)
        blk3_hn, b3_shape = _encode_block(blk2_hn, b2_shape, 256)
        blk4_hn, b4_shape = _encode_block(blk3_hn, b3_shape, 512)

        # downsampling + conv
        deblk4 = []
        deblk4.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        shape = valid_nd(b4_shape, kernel_size=(2,2), stride=(2,2))
        deblk4.append(Conv2D(num_filters=1024, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        deblk4.append(ELU())
        shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        deblk4.append(BatchNormalization())
        deblk4.append(Conv2D(num_filters=1024, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        deblk4.append(ELU())
        out_shape = same_nd(shape, kernel_size=(3,3), stride=(1,1))
        deblk4.append(BatchNormalization())
        # deconvolve
        deblk4.append(Conv2D_Transpose(num_filters=1024,
                                       kernel_size=(2,2), stride=(2,2), padding='SAME'))
        deblk4.append(ELU())
        deblk4.append(BatchNormalization())

        # decode and merge
        deblk4_hn = HiddenNode(prev=[blk4_hn], layers=deblk4)
        deblk3_hn = _merge_decode_block(deblk4_hn, blk4_hn, out_ch=256)
        deblk2_hn = _merge_decode_block(deblk3_hn, blk3_hn, out_ch=128)
        deblk1_hn = _merge_decode_block(deblk2_hn, blk2_hn, out_ch=64)

        # reduce channels
        blk = []
        blk.append(Conv2D(num_filters=32, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk.append(ELU())
        blk.append(BatchNormalization())
        blk.append(Conv2D(num_filters=16, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk.append(ELU())
        blk.append(BatchNormalization())
        deblk0_hn = HiddenNode(prev=[deblk1_hn, blk1_hn],
                                  input_merge_mode=Concat(axis=-1),
                                  layers=blk)

        self.endnode = EndNode(prev=[deblk0_hn])
