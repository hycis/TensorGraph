# Python2 compatibility
from __future__ import print_function

import numpy as np
import warnings
import tensorflow as tf

import sys
try:
    from tensorgraph.models_zoo.aibraintumormodel.nn.model \
        import model_C4R as model_module
except:
    warnings.warn("WARNING: Unable to locate model_C4R in TensorGraph package. Attempting relative path import")
    model_path = "../../../tensorgraph/models_zoo/"
    sys.path.append(model_path)
    from aibraintumormodel.nn.model import model_C4R as model_module

def test_model():
    with tf.Graph().as_default():
        image_size = (24, 128, 128, 1)
        rmics_size = (24, 128, 128, 8)
        batchsize = 4
        nclass0, nclass1 = (4,8)
        nseg = 2
        nclass_mat = np.zeros((nclass1, nclass0), dtype=np.float32)
        for i1 in range(nclass1):
            nclass_mat[i1, i1//2] = 1
        print("NCLASS_MAT:\n", nclass_mat)

        # Switches for training/testing
        training = tf.placeholder(tf.bool) # T/F for batchnorm
        kprob = tf.placeholder(tf.float32) # Dropout prob
        kprob_fcn_D1 = tf.placeholder(tf.float32)
        kprob_fcn_D2 = tf.placeholder(tf.float32)

        # Input placeholders
        t1_x = tf.placeholder(tf.float32, shape=(None,) + image_size, name='T1_ph')
        t2_x = tf.placeholder(tf.float32, shape=(None,) + image_size, name='T2_ph')
        tc_x = tf.placeholder(tf.float32, shape=(None,) + image_size, name='TC_ph')
        radiomics_x = tf.placeholder(tf.float32, shape=(None,) + rmics_size, \
            name='RADIOMICS_ph')

        # Model declaration
        model = model_module.Model(nseg_class=nseg, \
            nclass0=nclass0, nclass1=nclass1, nclass_mat=nclass_mat, \
            kprob=kprob, kprob_fcn_D1=kprob_fcn_D1, kprob_fcn_D2=kprob_fcn_D1)

        # Train
        yseg_train, ycls0_train, ycls1_train = model.train_fprop( \
            t1_x, t2_x, tc_x, radiomics_x)
        # Inference
        yseg_test, ycls0_test, ycls1_test = model.test_fprop( \
            t1_x, t2_x, tc_x, radiomics_x)

        y_pseg, y0_pred, y1_pred = tf.cond(training, \
            lambda: (yseg_train, ycls0_train, ycls1_train), \
            lambda: (yseg_test , ycls0_test , ycls1_test))

        # Initialize session
        config_proto = tf.ConfigProto(allow_soft_placement = True)
        config_proto.gpu_options.allow_growth = True
        #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.95
        config_proto.log_device_placement = False
        sess = tf.Session(config = config_proto)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Train

        t1_sb = np.random.rand(*((batchsize,) + image_size))
        t2_sb = np.random.rand(*((batchsize,) + image_size))
        tc_sb = np.random.rand(*((batchsize,) + image_size))
        rmics_sb = np.random.rand(*((batchsize,) + rmics_size))

        feedDictt = {training: True, kprob: 1.0, kprob_fcn_D1: 0.8, \
            kprob_fcn_D1: 0.8, t1_x: t1_sb, t2_x: t2_sb, tc_x: tc_sb, \
            radiomics_x: rmics_sb}
        feedDictv = {training: False, kprob: 1.0, kprob_fcn_D1: 1.0, \
            kprob_fcn_D1: 1.0, t1_x: t1_sb, t2_x: t2_sb, tc_x: tc_sb, \
            radiomics_x: rmics_sb}

        # Training graph
        y_pseg_sb, y0_pred_sb, y1_pred_sb = sess.run( \
            [y_pseg, y0_pred, y1_pred], feed_dict=feedDictt)
        print("TRAINING:")
        print("\tY_SEG  SHAPE: ", y_pseg_sb.shape)
        print("\tY0_CLS SHAPE: ", y0_pred_sb.shape)
        print("\tY1_CLS SHAPE: ", y1_pred_sb.shape)
        sys.stdout.flush()
        assert y_pseg_sb.shape == (batchsize,) + image_size[:-1] + (nseg,), \
            "ERROR: Wrong Y_SEG shape in output"
        assert y0_pred_sb.shape == (batchsize, nclass0), \
            "ERROR: Wrong Y0_CLS shape in output"
        assert y1_pred_sb.shape == (batchsize, nclass1), \
            "ERROR: Wrong Y1_CLS shape in output"

        # Inference graph
        y_pseg_sb, y0_pred_sb, y1_pred_sb = sess.run( \
            [y_pseg, y0_pred, y1_pred], feed_dict=feedDictv)
        print("INFERENCE:")
        print("\tY_SEG  SHAPE: ", y_pseg_sb.shape)
        print("\tY0_CLS SHAPE: ", y0_pred_sb.shape)
        print("\tY1_CLS SHAPE: ", y1_pred_sb.shape)
        sys.stdout.flush()
        assert y_pseg_sb.shape == (batchsize,) + image_size[:-1] + (nseg,), \
            "ERROR: Wrong Y_SEG shape in output"
        assert y0_pred_sb.shape == (batchsize, nclass0), \
            "ERROR: Wrong Y0_CLS shape in output"
        assert y1_pred_sb.shape == (batchsize, nclass1), \
            "ERROR: Wrong Y1_CLS shape in output"

if __name__ == '__main__':
    test_model()
