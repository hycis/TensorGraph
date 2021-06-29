from ....graph import Graph
from ....node import StartNode, HiddenNode, EndNode
from ....layers import MaxPooling, ELU, BatchNormalization, Concat, BaseModel, Sigmoid
from ....layers.conv import Conv2D_Transpose, Atrous_Conv2D, Conv2D
import numpy as np
class Dilated_Unet(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, nclasses=1):
        def downsampling_block(in_hn, filters):
            blk = []
            blk.append(Atrous_Conv2D(rate=1, num_filters=filters, kernel_size=(3,3), padding='SAME'))
            blk.append(BatchNormalization())
            blk.append(ELU())
            blk.append(Atrous_Conv2D(rate=2, num_filters=filters, kernel_size=(3,3), padding='SAME'))
            blk.append(BatchNormalization())
            blk.append(ELU())
            out_skip = HiddenNode(prev=[in_hn], layers=blk)
            out_hn = HiddenNode(prev=[out_skip], layers=[MaxPooling(stride=(2,2))])
            return out_hn, out_skip

        def upsampling_bolock(in_hn, in_skip, filters):
            blk = []
            blk.append(Conv2D(num_filters=filters, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(BatchNormalization())
            blk.append(ELU())
            blk.append(Conv2D(num_filters=filters, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(BatchNormalization())
            blk.append(ELU())
            out_hn = HiddenNode(prev=[in_hn, in_skip], input_merge_mode=Concat(axis=-1), layers=blk)
            return out_hn

        def dilation_block(in_hn, filters, dilation_rate):
            blk = []
            blk.append(Atrous_Conv2D(rate=dilation_rate, num_filters=filters, kernel_size=(3,3), padding='SAME'))
            blk.append(BatchNormalization())
            blk.append(ELU())
            out_hn = HiddenNode(prev=[in_hn], layers=blk)
            return out_hn

        self.startnode = StartNode(input_vars=[None])
        # encoding layers
        blk1_hn, blk1_skip = downsampling_block(self.startnode, 64)
        blk2_hn, blk2_skip = downsampling_block(blk1_hn, 128)
        blk3_hn, blk3_skip = downsampling_block(blk2_hn, 256)
        blk4_hn, blk4_skip = downsampling_block(blk3_hn, 512)
        # dilation layers
        dilation_rate = [1, 2, 4, 8, 16]
        dl1_hn = dilation_block(blk4_hn, 512, dilation_rate[0])
        dl2_hn = dilation_block(dl1_hn, 512, dilation_rate[1])
        dl3_hn = dilation_block(dl2_hn, 512, dilation_rate[2])
        dl4_hn = dilation_block(dl3_hn, 512, dilation_rate[3])
        dl5_hn = dilation_block(dl4_hn, 512, dilation_rate[4])
        # decoding layers
        pre4 = HiddenNode(prev=[dl5_hn], layers=[Conv2D_Transpose(kernel_size=(3,3), num_filters=512, stride=(2,2), padding='SAME')])
        deblk4_hn = upsampling_bolock(pre4, blk4_skip, 512)
        pre3 = HiddenNode(prev=[deblk4_hn], layers=[Conv2D_Transpose(kernel_size=(3,3), num_filters=256, stride=(2,2), padding='SAME')])
        deblk3_hn = upsampling_bolock(pre3, blk3_skip, 256)
        pre2 = HiddenNode(prev=[deblk3_hn], layers=[Conv2D_Transpose(kernel_size=(3,3), num_filters=128, stride=(2,2), padding='SAME')])
        deblk2_hn = upsampling_bolock(pre2, blk2_skip, 128)
        pre1 = HiddenNode(prev=[deblk2_hn], layers=[Conv2D_Transpose(kernel_size=(3,3), num_filters=64, stride=(2,2), padding='SAME')])
        deblk1_hn = upsampling_bolock(pre1, blk1_skip, 64)
        out_hn = HiddenNode(prev=[deblk1_hn], layers=[Conv2D(num_filters=nclasses, kernel_size=(1,1)), Sigmoid()])
        self.endnode = EndNode(prev=[out_hn])
        

