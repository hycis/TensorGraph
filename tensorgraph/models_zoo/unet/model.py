import tensorflow as tf
from ...node   import StartNode, HiddenNode, EndNode
from ...layers import BaseModel, BatchNormalization, RELU ,MaxPooling, \
                               Conv2D, Conv2D_Transpose, Dropout, Concat

class Convbnrelu(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self,kernel_size,stride,nfilters,drop=True):
        """
        define a model object.

        Args:
            kernel_size (tuple): Kernel size.
            stride (tuple): Stride size.
            nfilters (integer): The number of filters.
            drop (bool): whether dropout.

        """
        layers = []
        layers.append(Conv2D(num_filters=nfilters, kernel_size=kernel_size, stride=stride,padding='SAME'))
        layers.append(BatchNormalization())
        layers.append(RELU())
        if drop:
            layers.append(Dropout(0.02))
        self.startnode = StartNode(input_vars=[None])
        model_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[model_hn])


class Unet(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, nclass):
        """
        define a Unet model object.
        Paper: https://arxiv.org/abs/1505.04597

        Args:
            nclass (integer): The number of classes of the ouput mask.
            h (integer): The height of the input image.
            w (integer): The width of the input image.
            c (integer): The number of channels of the input image.
        """
        ksize = (3,3)
        stride = (1,1)
        filters = [32,64,128,256]
        poolsize = (2,2)
        poolstride = (2,2)

        #down1
        blk1 = []
        blk1.append(Convbnrelu(ksize,stride,filters[0]))
        blk1.append(Convbnrelu(ksize,stride,filters[0],False))
        maxpool1 =[]
        maxpool1.append(MaxPooling(poolsize=poolsize, stride=poolstride, padding='SAME'))

        #down2
        blk2 = []
        blk2.append(Convbnrelu(ksize,stride,filters[1]))
        blk2.append(Convbnrelu(ksize,stride,filters[1],False))
        maxpool2 =[]
        maxpool2.append(MaxPooling(poolsize=poolsize, stride=poolstride, padding='SAME'))

        #down3
        blk3 = []
        blk3.append(Convbnrelu(ksize,stride,filters[2]))
        blk3.append(Convbnrelu(ksize,stride,filters[2],False))
        maxpool3 = []
        maxpool3.append(MaxPooling(poolsize=poolsize, stride=poolstride, padding='SAME'))

        #down4
        blk4 = []
        blk4.append(Convbnrelu(ksize,stride,filters[3]))
        blk4.append(Convbnrelu(ksize,stride,filters[3],False))

        #up1
        transpose1 =[]
        transpose1.append(Conv2D_Transpose(filters[2], kernel_size=ksize, stride=poolstride,
                  padding='SAME'))

        blk5 = []
        blk5.append(Convbnrelu(ksize,stride,filters[2]))
        blk5.append(Convbnrelu(ksize,stride,filters[2],False))

        #up2
        transpose2 =[]
        transpose2.append(Conv2D_Transpose(filters[1], kernel_size=ksize, stride=poolstride,
                  padding='SAME'))
        blk6 = []
        blk6.append(Convbnrelu(ksize,stride,filters[1]))
        blk6.append(Convbnrelu(ksize,stride,filters[1],False))

        #up3
        transpose3 =[]
        transpose3.append(Conv2D_Transpose(filters[0], kernel_size=ksize, stride=poolstride,
                  padding='SAME'))
        blk7 = []
        blk7.append(Convbnrelu(ksize,stride,filters[0]))
        blk7.append(Convbnrelu(ksize,stride,filters[0],False))
        blk7.append(Conv2D(num_filters=nclass,kernel_size=(1,1),stride=stride,padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        blk1_hn = HiddenNode(prev=[self.startnode], layers=blk1)
        maxpool1_hn = HiddenNode(prev=[blk1_hn],layers=maxpool1)
        blk2_hn = HiddenNode(prev=[maxpool1_hn], layers=blk2)
        maxpool2_hn = HiddenNode(prev=[blk2_hn],layers=maxpool2)
        blk3_hn = HiddenNode(prev=[maxpool2_hn], layers=blk3)
        maxpool3_hn = HiddenNode(prev=[blk3_hn],layers=maxpool3)
        blk4_hn = HiddenNode(prev=[maxpool3_hn], layers=blk4)
        transpose1_hn = HiddenNode(prev=[blk4_hn], layers=transpose1)
        up1_hn = HiddenNode(prev=[transpose1_hn,blk3_hn],input_merge_mode=Concat(axis=-1),layers=blk5)
        transpose2_hn = HiddenNode(prev=[up1_hn], layers=transpose2)
        up2_hn = HiddenNode(prev=[transpose2_hn,blk2_hn],input_merge_mode=Concat(axis=-1),layers=blk6)
        transpose3_hn = HiddenNode(prev=[up2_hn], layers=transpose3)
        up3_hn = HiddenNode(prev=[transpose3_hn,blk1_hn],input_merge_mode=Concat(axis=-1),layers=blk7)
        self.endnode = EndNode(prev=[up3_hn])
