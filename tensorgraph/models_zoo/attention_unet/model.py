
from ...graph import Graph
from ...node import StartNode, HiddenNode, EndNode
from ...layers import Conv2D, MaxPooling, RELU, ELU, Sigmoid, BatchNormalization, Sum, Multiply, \
                      Concat, AvgPooling, Conv2D_Transpose, Dropout, BaseModel
import numpy as np

class Attention_UNet(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, input_shape):

        def _encode_block(in_hn, out_ch):
            blk = []
            blk.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
            blk.append(Conv2D(num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization())
            blk.append(Conv2D(num_filters=out_ch, kernel_size=(3,3), stride=(1,1), padding='SAME'))
            blk.append(ELU())
            blk.append(BatchNormalization())
            out_hn = HiddenNode(prev=[in_hn], layers=blk)
            return out_hn

        def _attention_block(deblk_hn, blk_hn, out_ch):
            layers1 = []
            layers2 = []
            layers1.append(Conv2D(num_filters=out_ch, kernel_size=(1,1), stride=(1,1), padding='SAME'))
            layers1.append(BatchNormalization())
            layers2.append(Conv2D(num_filters=out_ch, kernel_size=(1,1), stride=(1,1), padding='SAME'))
            layers2.append(BatchNormalization())
            hn1 = HiddenNode(prev=[deblk_hn], layers=layers1)
            hn2 = HiddenNode(prev=[blk_hn], layers=layers2)
            layers3 = []
            layers3.append(RELU())
            layers3.append(Conv2D(num_filters=1, kernel_size=(1,1), stride=(1,1), padding='SAME'))
            layers3.append(BatchNormalization())
            layers3.append(Sigmoid())
            hn3 = HiddenNode(prev=[hn1,hn2], input_merge_mode=Sum(), layers=layers3)
            return HiddenNode(prev=[hn3, blk_hn], input_merge_mode=Multiply(), layers=[])

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

            att_hn = _attention_block(deblk_hn, blk_hn, out_ch)
            out_hn = HiddenNode(prev=[att_hn, blk_hn],
                                   input_merge_mode=Concat(axis=-1),
                                   layers=blk)
            return out_hn


        # encoding
        blk1 = []
        blk1.append(Conv2D(num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk1.append(ELU())
        blk1.append(BatchNormalization())
        blk1.append(Conv2D(num_filters=64, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        blk1.append(ELU())
        blk1.append(BatchNormalization())

        self.startnode = StartNode(input_vars=[None])
        blk1_hn = HiddenNode(prev=[self.startnode], layers=blk1)
        blk2_hn = _encode_block(blk1_hn, 128)
        blk3_hn = _encode_block(blk2_hn, 256)
        blk4_hn = _encode_block(blk3_hn, 512)

        # downsampling + conv
        deblk4 = []
        deblk4.append(MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'))
        deblk4.append(Conv2D(num_filters=1024, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        deblk4.append(ELU())
        deblk4.append(BatchNormalization())
        deblk4.append(Conv2D(num_filters=1024, kernel_size=(3,3), stride=(1,1), padding='SAME'))
        deblk4.append(ELU())
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
