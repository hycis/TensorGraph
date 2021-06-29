'''
#-------------------------------------------------------------------------------
Component model for API, TensorGraph implementation
Louis Lee 07-02-2019
#-------------------------------------------------------------------------------
API details:
    Version: ML_BRAIN_TUMOR_v2.0.4b
    Internal identifier: Model5b
Compnent model details:
    Version: v1.0.0b
    Internal identifier: Model_C5XSb
Notes:
    Checkpoints from v2.0.4 are incompatible with this version
    Actual model implementation may vary due to slightly different sequence
    ordering of layers
#-------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

###############################################################################
# Model C5XS - 5 levels, further reduced clsblk size & cls conv channels      #
# cls1 feeds to cls0 through a ReLU s.t. input to cls0 is >= 0                #
###############################################################################
import tensorflow as tf
from .....graph import Graph
from .....node import StartNode, HiddenNode, EndNode
from .....layers import BaseLayer, BaseModel, Flatten, Multiply, Concat, NoChange
from .CommonBlocks import *

class Model(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, nseg_class, nclass0, nclass1, nclass_mat, \
        kprob, kprob_fcn_D1, kprob_fcn_D2, class_name=None, \
        bn_flavor='TFBatchNorm'):

        # Static  variables
        ksize_111 = (1,1,1)
        ksize_133 = (1,3,3)
        ksize_333 = (3,3,3)
        ksize_244 = (2,4,4)
        ksize_444 = (4,4,4)
        ksize_355 = (3,5,5)
        ksize_555 = (5,5,5)
        stride_111 = (1,1,1)
        stride_122 = (1,2,2)
        stride_211 = (2,1,1)
        stride_222 = (2,2,2)
        filters_seg = [6,8,12,24,48,96,128,192]
        filters_cls = [4,4, 8,12,24,48, 64, 96]
        filters_fcn = [192,384]

        self.startnode = StartNode(input_vars=[None])
        with tf.name_scope("model_input"):
            blkD_hn = HiddenNode(prev=[self.startnode], input_merge_mode=Concat(axis=-1))

        with tf.name_scope("model_seg"):
            blk01_hn = HiddenNode(prev=[blkD_hn], layers=[ \
                SegConvV2(num_filters1=filters_seg[1], num_filters2=filters_seg[1], \
                kernel_size1=ksize_355, kernel_size2=ksize_133, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_333, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk02_hn = HiddenNode(prev=[blk01_hn], layers=[ \
                SegConvV2(num_filters1=filters_seg[2], num_filters2=filters_seg[2], \
                kernel_size1=ksize_355, kernel_size2=ksize_133, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_333, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk03_hn = HiddenNode(prev=[blk02_hn], layers=[ \
                SegConvV2(num_filters1=filters_seg[3], num_filters2=filters_seg[3], \
                kernel_size1=ksize_355, kernel_size2=ksize_133, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_333, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk04_hn = HiddenNode(prev=[blk03_hn], layers=[ \
                SegConvV2(num_filters1=filters_seg[4], num_filters2=filters_seg[4], \
                kernel_size1=ksize_355, kernel_size2=ksize_333, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_333, pool_stride=stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk05_hn = HiddenNode(prev=[blk04_hn], layers=[ \
                ConvV2(num_filters=filters_seg[5], \
                kernel_size=ksize_333, stride=stride_111, \
                pool_size=ksize_333, pool_stride=stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])

            blk06_hn = HiddenNode(prev=[blk05_hn], layers=[ \
                SegDeconvV2(num_filters=filters_seg[5], kernel_size=ksize_444, \
                kernel_stride=stride_222, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk07_hn = HiddenNode(prev=[blk06_hn, blk04_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV2(num_filters=filters_seg[4], kernel_size=ksize_444, \
                kernel_stride=stride_222, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk08_hn = HiddenNode(prev=[blk07_hn, blk03_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV2(num_filters=filters_seg[3], kernel_size=ksize_244, \
                kernel_stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk09_hn = HiddenNode(prev=[blk08_hn, blk02_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV2(num_filters=filters_seg[2], kernel_size=ksize_244, \
                kernel_stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk10_hn = HiddenNode(prev=[blk09_hn, blk01_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV2(num_filters=filters_seg[1], kernel_size=ksize_244, \
                kernel_stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            outseg_hn = HiddenNode(prev=[blk10_hn], layers=[ \
                SegOutput(out_nchn=nseg_class, \
                num_filters=filters_seg[0], kernel_size=ksize_333, \
                kernel_stride=stride_111, keep_prob=kprob)])

        with tf.name_scope("model_cls"):
            # Classification
            outmsk_hn = HiddenNode(prev=[outseg_hn], layers=[MaskSoftmaxIdentity()])
            redmsk_hn = HiddenNode(prev=[outmsk_hn], layers=[MaskReduceDim()])
            attmsk_hn = HiddenNode(prev=[redmsk_hn, blkD_hn], input_merge_mode=Multiply())

            blk11_hn = HiddenNode(prev=[blkD_hn, attmsk_hn, outmsk_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV4(num_filters=filters_cls[0], kernel_size=ksize_133, \
                pool_size=ksize_333, pool_stride = stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            outmskD1_hn = HiddenNode(prev=[outmsk_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_133, stride=stride_122, padding='SAME')])
            redmskD1_hn = HiddenNode(prev=[redmsk_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_133, stride=stride_122, padding='SAME')])

            attmskD1_hn = HiddenNode(prev=[redmskD1_hn, blk11_hn], input_merge_mode=Multiply())
            blk12_hn = HiddenNode(prev=[blk11_hn, attmskD1_hn, outmskD1_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV4(num_filters=filters_cls[1], kernel_size=ksize_133, \
                pool_size=ksize_333, pool_stride = stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            outmskD2_hn = HiddenNode(prev=[outmskD1_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_133, stride=stride_122, padding='SAME')])
            redmskD2_hn = HiddenNode(prev=[redmskD1_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_133, stride=stride_122, padding='SAME')])

            attmskD2_hn = HiddenNode(prev=[redmskD2_hn, blk12_hn], input_merge_mode=Multiply())
            blk13_hn = HiddenNode(prev=[blk12_hn, attmskD2_hn, outmskD2_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV4(num_filters=filters_cls[2], kernel_size=ksize_133, \
                pool_size=ksize_333, pool_stride = stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            outmskD3_hn = HiddenNode(prev=[outmskD2_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_133, stride=stride_122, padding='SAME')])
            redmskD3_hn = HiddenNode(prev=[redmskD2_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_133, stride=stride_122, padding='SAME')])

            attmskD3_hn = HiddenNode(prev=[redmskD3_hn, blk13_hn], input_merge_mode=Multiply())
            blk14_hn = HiddenNode(prev=[blk13_hn, attmskD3_hn, outmskD3_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV4(num_filters=filters_cls[3], kernel_size=ksize_133, \
                pool_size=ksize_333, pool_stride = stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            outmskD4_hn = HiddenNode(prev=[outmskD3_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_333, stride=stride_222, padding='SAME')])
            redmskD4_hn = HiddenNode(prev=[redmskD3_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_333, stride=stride_222, padding='SAME')])

            attmskD4_hn = HiddenNode(prev=[redmskD4_hn, blk14_hn], input_merge_mode=Multiply())
            blk15_hn = HiddenNode(prev=[blk14_hn, attmskD4_hn, outmskD4_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV4(num_filters=filters_cls[4], kernel_size=ksize_333, \
                pool_size=ksize_333, pool_stride = stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])

            blk16_hn = HiddenNode(prev=[blk15_hn], layers=[ \
                Conv3D(num_filters=filters_fcn[0], kernel_size=ksize_333, \
                stride=stride_111, padding='SAME')])
            blk16_hn = HiddenNode(prev=[blk16_hn, blk05_hn], input_merge_mode=Concat(axis=-1))
            blk16_hn = HiddenNode(prev=[blk16_hn], layers=[ \
                MaxPooling3D(poolsize=ksize_333, stride=stride_222, padding='SAME')])

        with tf.name_scope("model_fcn"):
            outcls_hn = HiddenNode(prev=[blk15_hn], layers=[ \
                ClsFCNV2(fcn1_dim=filters_fcn[1], fcn1_keep_prob=kprob_fcn_D1, \
                nclass1=nclass1, nclass0=nclass0, nclass_mat=nclass_mat)])

        self.endnode = EndNode(prev=[outseg_hn, outcls_hn], input_merge_mode=NoChange())

    def train_fprop(self, t1, t2, tc, rmics=None):
        print("Model C5XS")
        assert rmics is None, "ERROR: Radiomics not used in this model"
        out_seg, out_cls = super(Model, self).train_fprop(t1, t2, tc)
        # Returns seg, cls0, cls1
        return (out_seg,) + tuple(out_cls)

    def test_fprop(self, t1, t2, tc, rmics=None):
        assert rmics is None, "ERROR: Radiomics not used in this model"
        out_seg, out_cls = super(Model, self).test_fprop(t1, t2, tc)
        # Returns seg, cls0, cls1
        return (out_seg,) + tuple(out_cls)
