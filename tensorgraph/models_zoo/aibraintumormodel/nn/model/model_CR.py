'''
#-------------------------------------------------------------------------------
Component model for API
Louis Lee 11-10-2018
#-------------------------------------------------------------------------------
API details:
    Version: ML_BRAIN_TUMOR_v2.0.0
    Internal identifier: Model5
Compnent model details:
    Version: v1.0.0
    Internal identifier: Model_CR
#-------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

###############################################################################
# Model CR                                                                    #
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
        ksize_355 = (3,5,5)
        ksize_244 = (2,4,4)
        ksize_444 = (4,4,4)
        ksize_555 = (5,5,5)
        stride_111 = (1,1,1)
        stride_122 = (1,2,2)
        stride_211 = (2,1,1)
        stride_222 = (2,2,2)
        filters = [4,8,16,32,64,128,256]

        self.startnode = StartNode(input_vars=[None])
        with tf.name_scope("model_input"):
            blkD_hn = HiddenNode(prev=[self.startnode], input_merge_mode=Concat(axis=-1))

        with tf.name_scope("model_seg"):
            blk01_hn = HiddenNode(prev=[blkD_hn], layers=[ \
                ConvV1(num_filters=filters[2], kernel_size=ksize_355, \
                stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk02_hn = HiddenNode(prev=[blk01_hn], layers=[ \
                SegConvV1(num_filters1=filters[2], num_filters2=filters[2], \
                kernel_size1=ksize_355, kernel_size2=ksize_133, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_133, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk03_hn = HiddenNode(prev=[blk02_hn], layers=[ \
                SegConvV1(num_filters1=filters[3], num_filters2=filters[3], \
                kernel_size1=ksize_355, kernel_size2=ksize_133, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_133, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk04_hn = HiddenNode(prev=[blk03_hn], layers=[ \
                SegConvV1(num_filters1=filters[4], num_filters2=filters[4], \
                kernel_size1=ksize_355, kernel_size2=ksize_333, \
                kernel_stride1=stride_111, kernel_stride2=stride_111, \
                pool_size=ksize_333, pool_stride=stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])

            blk05_hn = HiddenNode(prev=[blk04_hn], layers=[ \
                SegDeconvV1(num_filters=filters[4], kernel_size=ksize_444, \
                kernel_stride=stride_222, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk06_hn = HiddenNode(prev=[blk05_hn, blk03_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV1(num_filters=filters[3], kernel_size=ksize_244, \
                kernel_stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk07_hn = HiddenNode(prev=[blk06_hn, blk02_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV1(num_filters=filters[2], kernel_size=ksize_244, \
                kernel_stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            blk08_hn = HiddenNode(prev=[blk07_hn, blk01_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                SegDeconvV1(num_filters=filters[1], kernel_size=ksize_244, \
                kernel_stride=stride_122, keep_prob=kprob, bn_flavor=bn_flavor)])
            outseg_hn = HiddenNode(prev=[blk08_hn], layers=[ \
                SegOutput(out_nchn=nseg_class, \
                num_filters=filters[0], kernel_size=ksize_333, \
                kernel_stride=stride_111, keep_prob=kprob)])

        with tf.name_scope("model_cls"):
            # Classification
            outmsk_hn = HiddenNode(prev=[outseg_hn], layers=[MaskSoftmaxIdentity()])
            redmsk_hn = HiddenNode(prev=[outmsk_hn], layers=[MaskReduceDim(), MaskSquare()])
            attmsk_hn = HiddenNode(prev=[redmsk_hn, blkD_hn], input_merge_mode=Multiply())

            blk10_hn = HiddenNode(prev=[blkD_hn, attmsk_hn, outmsk_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV1(num_filters=filters[0], kernel_size=ksize_333, \
                pool_size=ksize_333, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk11_hn = HiddenNode(prev=[blk10_hn, blk07_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV1(num_filters=filters[1], kernel_size=ksize_333, \
                pool_size=ksize_133, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk12_hn = HiddenNode(prev=[blk11_hn, blk06_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV1(num_filters=filters[3], kernel_size=ksize_333, \
                pool_size=ksize_333, pool_stride=stride_122, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk13_hn = HiddenNode(prev=[blk12_hn, blk05_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV1(num_filters=filters[3], kernel_size=ksize_333, \
                pool_size=ksize_333, pool_stride=stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk14_hn = HiddenNode(prev=[blk13_hn, blk04_hn], \
                input_merge_mode=Concat(axis=-1), layers=[ \
                ClsDenseV1(num_filters=filters[4], kernel_size=ksize_333, \
                pool_size=ksize_333, pool_stride=stride_222, keep_prob=kprob, \
                bn_flavor=bn_flavor)])
            blk15_hn = HiddenNode(prev=[blk14_hn], layers=[ \
                ClsOutput(num_filters=filters[6], kernel_size=ksize_333, \
                kernel_stride=stride_111, pool_size=ksize_333, \
                pool_stride=stride_222, keep_prob=kprob)])

        with tf.name_scope("model_fcn"):
            outcls_hn = HiddenNode(prev=[blk15_hn], layers=[ \
                ClsFCNV1(fcn1_dim=filters[6], fcn1_keep_prob=kprob_fcn_D1, \
                    fcn2_dim=filters[5], fcn2_keep_prob=kprob_fcn_D2, \
                nclass1=nclass1, nclass0=nclass0, nclass_mat=nclass_mat)])

        self.endnode = EndNode(prev=[outseg_hn, outcls_hn], input_merge_mode=NoChange())

    def train_fprop(self, t1, t2, tc, rmics=None):
        print("Model CR")
        assert rmics is not None, "ERROR: Radiomics required in this model"
        out_seg, out_cls = super(Model, self).train_fprop(t1, t2, tc, rmics)
        # Returns seg, cls0, cls1
        return (out_seg,) + tuple(out_cls)

    def test_fprop(self, t1, t2, tc, rmics=None):
        assert rmics is not None, "ERROR: Radiomics required in this model"
        out_seg, out_cls = super(Model, self).test_fprop(t1, t2, tc, rmics)
        # Returns seg, cls0, cls1
        return (out_seg,) + tuple(out_cls)
