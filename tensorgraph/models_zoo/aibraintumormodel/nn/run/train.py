'''
#------------------------------------------------------------------------------
# Main train routines WLRRDDC
#------------------------------------------------------------------------------
# W - Water segmentation
# L - Cyclical learning rate
# R - Radiomics
# R - Batch renormalization
# D - Series dropout (from dataset)
# D - Distributed batch normalization
# C - New costAccumulator (fixed some summation bugs & cleanup)
#------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorgraph as tg
import horovod.tensorflow as hvd
import sys
import random
import itertools
import pickle
import sklearn.metrics
import time
import collections
import os
import importlib

# MPI
import mpi4py.rc
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import mpi4py.MPI as MPI

from . import costfunction as costfunction
from . import radiomicsFeatures as radiomicsFeatures

class trainModel:
    '''
    Object to coordinate model training
    Governs data & model initialization, training & validation loops
    Called by main script to control & run training
    '''
    def __init__(self, data_module, model_module, model_scope, batchsize, \
        weight_decay, batch_renorm_rmax_dmax, anatomies, biopsy_only, \
        mask_only, water_mask, series_kprob, train_valid_seed, \
        validation_list, radiomics, clsmatcoeff, distributed_batchnorm, \
        data_dir, testing):
        '''
        Initialization
        Inputs:
            data_module - Python module for TFRecord data pipeline
            model_module - Python module for model script
            model_scope - String for top-level model scope name
            batchsize - Training batch size
            weight_decay - T/F whether to use L2 regularization in training
            batch_renorm_rmax_dmax - tuple of (rmax,dmax) for batch renormalization
            anatomies - list of anatomies to be selected in data_object's
                        pipeline (defaults to [] i.e. all)
            biopsy_only - T/F whether to use only biopsy-confirmed data only
            mask_only - T/F whether to use only data with segmentation mask
            water_mask - T/F whether segmentation mask should have 3 channels
                         (T) with (BG/tumor/water) or 2 channels (F) with
                         (BG/tumor) where tumor mask is combined with water
            series_kprob - tuple of (t1,t2,tc,dwi) containing probability of
                           each series being kept (i.e. not zerorized) during
                           training
            train_valid_seed - seed to feed into TF's RNG to generate
                               train/validation dataset split
            validation_list - path to text file containing newline-delimited
                              absolute paths to TFRecord files that are to be
                              used as validation set. If set, train_valid_seed
                              is ignored
            radiocmis - T/F whether to use radiomics as another input channel
                        to model
            clsmatcoeff - FP tuple of (neg,pos) for coefficients of small-to-big
                          class relationship/adjacency matrix. neg = weight
                          between unrelated classes, pos = weight between
                          related classes (i.e. when small cls is a subclass
                          of big cls)
            distributed_batchnorm - T/F whether to use distributed batchnorm
            data_root_dir - dir of TFRecords
            testing - flag that changes some logic for testing purposes
        '''
        # Model setup parameters
        self.model_module = importlib.import_module(model_module)
        # Import data module
        data_module = importlib.import_module(data_module)

        # Initialize training session
        self.data = data_module.Data(root_dir=data_dir, anatomies=anatomies, \
            biopsy_only=biopsy_only, mask_only=mask_only, water_mask=water_mask, \
            series_kprob=series_kprob, train_valid_seed=train_valid_seed, \
            valid_list=validation_list, clsmatcoeff=clsmatcoeff, testing=testing)

        self.runModel = runModel(data_object=self.data, model_module=self.model_module, \
            batchsize=batchsize, weight_decay=weight_decay, model_scope=model_scope, \
            batch_renorm_rmax_dmax=batch_renorm_rmax_dmax, radiomics=radiomics, \
            distributed_batchnorm=distributed_batchnorm, testing=testing)

    def configureTraining(self, flags, save_path, restore_path, log_path):
        '''
        Internal script to configure training environment
        - Sets parameters
        - Initializes training session, variables, and methods
        - Restores model
        Inputs: flags - configReader object instance containing training parameters
                save_path - path to save checkpoint (e.g. "/path/to/tfgraph.ckpt")
                restore_path - path from which to restore checkpoint (e.g. "/path/to/tfgraph.ckpt")
                log_path - path to save tensorboard summary (e.g. "/path/to/log/")
        '''
        self.flags = flags
        self.save_path = save_path
        self.restore_path = restore_path
        self.log_path = log_path

        self.save_out_res = self.flags.out_res_path is not None

        self.runModel.initializeTraining(self.flags, self.log_path)

        # Restore model
        if self.flags.restore:
            self.runModel.restoreModel(self.restore_path)

    def train(self):
        '''
        Internal script to control training & validation cycles
        - Trains model
        - Validates model
        - Saves trained model
        Inputs: None
        '''
        iepoch = 0
        while (iepoch < self.flags.max_epochs):
            # Train one epoch
            self.runModel.trainOneEpoch(iepoch, \
                self.flags.save, self.save_path, \
                self.flags.save_every_nsteps, self.flags.report_every_nsteps)

            # Validate every certain number of epochs
            if (self.flags.validate_every_nepoch > 0 and \
                iepoch % self.flags.validate_every_nepoch == 0) or \
                iepoch == self.flags.max_epochs-1:
                self.runModel.validateOneEpoch(iepoch,
                    self.flags.save, self.save_path, \
                    self.flags.out_res_path, self.flags.sel_threshold, \
                    iepoch%self.flags.save_out_every_nepoch == 0)

            iepoch += 1
            MPI.COMM_WORLD.Barrier()

        # Save final output
        if self.flags.save:
            self.runModel.saveModel(self.save_path)

    def finalize(self):
        '''
        Internal script to finalize training
        - Terminates TF session
        Inputs: None
        '''
        self.runModel.finalize()

class runModel:
    '''
    Internal object containing actual training & validation implementation
    Called by an instance of trainModel()
    '''
    # Initialize model & data
    def __init__(self, data_object, model_module, batchsize, weight_decay, \
        model_scope=None, batch_renorm_rmax_dmax=None, radiomics=False, \
        distributed_batchnorm=False, testing=False):
        '''
        Variables declaration
        - Sets training parameters
        - Initializes data_object's pipeline & model
        - Declares namescopes
        - Defines placeholders
        '''
        self.__training_initialized = False
        tf.reset_default_graph()

        self.testing = testing

        assert batchsize > 0, "Error: batchsize must be >0"
        self.batchsize = batchsize
        self.model_module = model_module
        if model_scope is None:
            self.model_scope = ""
        else:
            self.model_scope = model_scope
        self.data_object = data_object
        self.outres_size = 40
        self.weight_decay = weight_decay
        if batch_renorm_rmax_dmax is None:
            self.batch_renorm = False
        else:
            self.batch_renorm = True
            renorm_rmax_start, renorm_dmax_start = batch_renorm_rmax_dmax
        self.radiomics = radiomics
        self.distributed_batchnorm = distributed_batchnorm

        # Make sure Horovod & MPI gives same # nodes
        assert hvd.size() == MPI.COMM_WORLD.Get_size()

        # TFRecord iterator
        with tf.device('/cpu:0'):
            with tf.name_scope('TFRecord'):
                self.train_iterator = self.data_object.generateBatch('train', \
                    self.batchsize, 4*self.batchsize, shuffle_batch=True, \
                    num_shards=hvd.size(), worker_rank=hvd.rank(), \
                    repeat=-1, prefetch_gpu=True)
                self.valid_iterator = self.data_object.generateBatch('valid', \
                    1, 1, shuffle_batch=False, \
                    num_shards=hvd.size(), worker_rank=hvd.rank(), \
                    repeat=1, prefetch_gpu=False)
            self.train_next = self.train_iterator.get_next()
            self.valid_next = self.valid_iterator.get_next()

        # Data size info required for graph construction
        self.origsize = self.data_object.getOrigSize()
        self.cropsize = self.data_object.getCropSize()
        self.masksize = self.data_object.getMaskSize()
        self.nloc = self.data_object.getNLoc()
        self.nclass0 = self.data_object.getOutputClasses0()
        self.nclass1 = self.data_object.getOutputClasses1()
        self.nseg = self.data_object.getOutputSegClasses()
        if self.radiomics:
            self.radiomics_size = \
                radiomicsFeatures.getNumFeatures(self.cropsize, nchannels=2)

        # TF Placeholders
        with tf.name_scope('InputPlaceHolder'):
            self.t1_in = tf.placeholder(tf.float32, \
                shape=(None,) + self.origsize, name='T1_ph')
            self.t2_in = tf.placeholder(tf.float32, \
                shape=(None,) + self.origsize, name='T2_ph')
            self.tc_in = tf.placeholder(tf.float32, \
                shape=(None,) + self.origsize, name='TC_ph')

            self.dwi_in = tf.placeholder(tf.float32, \
                shape=(None,) + self.origsize, name='DWI_ph')

            self.heat_in = tf.placeholder(tf.int32, \
                shape=(None,) + self.origsize, name='HEAT_ph')
            self.wheat_in = tf.placeholder(tf.int32, \
                shape=(None,) + self.origsize, name='WHEAT_ph')
            self.label_in = tf.placeholder(tf.int32, \
                shape=(None,), name='LABEL_ph')
            self.examno_in = tf.placeholder(tf.int32, \
                shape=(None,), name='EXAMNO_ph')
            self.age_in = tf.placeholder(tf.int32, \
                shape=(None,), name='AGE_ph')
            self.sex_in = tf.placeholder(tf.int32, \
                shape=(None,), name='SEX_ph')
            self.loc_in = tf.placeholder(tf.int32, \
                shape=(None, self.nloc), name='LOC_ph')

            # Switches for training/testing
            self.training = tf.placeholder(tf.bool) # T/F for batchnorm
            self.kprob = tf.placeholder(tf.float32) # Dropout prob
            self.kprob_fcn_D1 = tf.placeholder(tf.float32)
            self.kprob_fcn_D2 = tf.placeholder(tf.float32)

        with tf.name_scope('ModelInput'):
            if self.radiomics:
                # Direct input into model
                self.radiomics_x = tf.placeholder(tf.float32, \
                    shape=(None,) + self.radiomics_size, name='RADIOMICS_PH')
            else:
                self.radiomics_x = None

        if hvd.rank() == 0:
            print("Declare model")
        # Pass data through augmentation
        with tf.name_scope('DataAugPrep'):
            self.t1_x, self.t2_x, self.tc_x, self.dwi_x, \
            self.y_seg, self.y0_act, self.y1_act, \
            self.age_x, self.sex_x, self.channels_x = \
                self.data_object.batchAugPrep( \
                self.t1_in, self.t2_in, self.tc_in, self.dwi_in, \
                self.heat_in, self.wheat_in, self.label_in, \
                self.age_in, self.sex_in, self.training)
            self.y_examno, self.y_loc = (self.examno_in, self.loc_in)
            self.y_seg = tf.cast(self.y_seg, tf.float32)
            self.y0_act = tf.cast(self.y0_act, tf.float32)
            self.y1_act = tf.cast(self.y1_act, tf.float32)

        if self.batch_renorm:
            self.renorm_rmax = tf.Variable(renorm_rmax_start, \
                trainable=False, name='renorm_rmax', dtype=tf.float32)
            self.renorm_dmax = tf.Variable(renorm_dmax_start, \
                trainable=False, name='renorm_dmax', dtype=tf.float32)
            self.renorm_clipping = {'rmin': 1.0/self.renorm_rmax, \
                'rmax': self.renorm_rmax, 'dmax': self.renorm_dmax}
            self.renorm_rmin = self.renorm_clipping['rmin']
        else:
            self.renorm_clipping = None

        # Output from model
        if self.model_scope != "":
            self.model_module.Model.__name__ = self.model_scope
        self.model = self.model_module.Model( \
            self.nseg, self.nclass0, self.nclass1, self.data_object.getNClassMat(), \
            self.kprob, self.kprob_fcn_D1, self.kprob_fcn_D2)

        yseg_train, ycls0_train, ycls1_train = self.model.train_fprop( \
            self.t1_x, self.t2_x, self.tc_x, self.radiomics_x)
        yseg_test,  ycls0_test , ycls1_test  = self.model.test_fprop( \
            self.t1_x, self.t2_x, self.tc_x, self.radiomics_x)
        self.y_pseg, self.y0_pred, self.y1_pred = tf.cond(self.training, \
            lambda: (yseg_train, ycls0_train, ycls1_train), \
            lambda: (yseg_test , ycls0_test , ycls1_test))

        self.y0_pred_sfmax = tf.nn.softmax(self.y0_pred, axis=-1)
        self.y1_pred_sfmax = tf.nn.softmax(self.y1_pred, axis=-1)
        sys.stdout.flush()

        # Collect all model variables prior to declaring gradients in graph
        if hvd.rank() == 0:
            self.model_input = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                scope=os.path.join(self.model_scope,'model_input/'))
            self.model_seg = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                scope=os.path.join(self.model_scope,'model_seg/'))
            self.model_cls = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                scope=os.path.join(self.model_scope,'model_cls/'))
            self.model_fcn = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, \
                scope=os.path.join(self.model_scope,'model_fcn/'))

        MPI.COMM_WORLD.Barrier()

    # Set up training parameters
    def initializeTraining(self, flags, log_path):
        '''
        Methods initialization
        - Defines loss functions & minimization methods & optimization scope
        - Defines session attributes
        - Defines save/restore functions
        - Initializes session & session variables
        '''
        assert not self.__training_initialized, \
            "ERROR: Training parameters already initialized"
        self.flags = flags

        if self.flags.continuation:
            assert self.flags.restore, \
                "Error: restore must be T if continuation = T"

        assert (self.flags.keep_prob > 0 and self.flags.keep_prob <= 1.0), \
            "Error: kprob must be between (0,1.0]"

        # Optimizers
        optimizer_dict = { \
            'sgd' : tf.train.GradientDescentOptimizer, \
            'adam': tf.train.AdamOptimizer, \
        }
        assert self.flags.optimizer in optimizer_dict.keys(), \
            "ERROR: invalid optimizer"

        # sess.run() configs
        config_proto = tf.ConfigProto(allow_soft_placement = True)
        config_proto.gpu_options.allow_growth = True
        #config_proto.gpu_options.per_process_gpu_memory_fraction = 0.95
        config_proto.log_device_placement = False
        config_proto.gpu_options.visible_device_list = str(hvd.local_rank())

        if hvd.rank() == 0:
            print("HVD DEVICES : ", \
                config_proto.gpu_options.visible_device_list)
            print("HVD # NODES : ", hvd.size())

            print("KEEP_PROB     : ", self.flags.keep_prob)
            print("KEEP_PROB_FCN : ", self.flags.keep_prob_fcn_D1, \
                ", ", self.flags.keep_prob_fcn_D2)
            print("BATCH_SIZE    : ", self.batchsize)
            print("MIN_LRN_RATE  : ", self.flags.min_learning_rate)
            print("MAX_LRN_RATE  : ", self.flags.max_learning_rate)
            print("LRN_EPCOHSIZE : ", self.flags.learning_rate_epochsize)
            print("LRN_DECAY     : ", (self.flags.learning_rate_decay_step, \
                self.flags.learning_rate_decay_rate))
            print("WEIGHT_DECAY  : ", self.weight_decay)
            print("BATCH_RENORM  : ", self.batch_renorm)
            print("RADIOMICS     : ", self.radiomics)
            print("DISTRIBUTED_BN: ", self.distributed_batchnorm)
            print("TRAIN_SEG     : ", self.flags.train_seg)
            print("TRAIN_CLS     : ", self.flags.train_cls)
            print("BIG_CLS       : ", self.flags.big_cls)
            print("SMALL_CLS     : ", self.flags.small_cls)
        np.set_printoptions(linewidth=256)

        # # training data
        ntrain = self.data_object.getDataSize('train')
        # cls weights
        lweights0 = self.data_object.getLossClsWeights0()
        lweights1 = self.data_object.getLossClsWeights1()
        # seg weights squared
        sweights = np.asarray(self.data_object.getLossSegWeights())
        sweights2 = sweights*sweights
        sweights2 = 1.0*sweights2/np.sum(sweights2)

        if hvd.rank() == 0:
            print("# CLS CLASSES 0: ", self.nclass0)
            print("# CLS CLASSES 1: ", self.nclass1)
            print("# CLS WEIGHTS 0: ", len(lweights0))
            print("# CLS WEIGHTS 1: ", len(lweights1))
            print("# SEG WEIGHTS  : ", len(sweights2))
            print("SEG^2 WEIGHTS  : ", sweights2)

            print("TRAIN SIZE : ", self.data_object.getDataSize('train'))
            print("VALID SIZE : ", self.data_object.getDataSize('valid'))

            self.data_object.getDataCount()

            print("\nVALIDATION LIST:")
            self.data_object.listDataFiles('valid')

        # Training vars
        self.train_accu_cls0 = \
            costfunction.accuracy(self.y0_act, self.y0_pred, axis=-1)
        self.train_accu_cls1 = \
            costfunction.accuracy(self.y1_act, self.y1_pred, axis=-1)

        self.train_xent_cls0 = costfunction.weightedFocalCrossEntropyLoss( \
            self.y0_act, self.y0_pred, \
            weighting=True, weights=lweights0, focal=False, gamma=1)
        self.train_xent_cls1 = costfunction.weightedFocalCrossEntropyLoss( \
            self.y1_act, self.y1_pred, \
            weighting=True, weights=lweights1, focal=False, gamma=1)

        self.train_accu_seg = \
            costfunction.accuracy(self.y_seg, self.y_pseg, axis=-1)
        self.train_xent_seg = costfunction.weightedFocalCrossEntropyLoss( \
            self.y_seg, self.y_pseg, \
            weighting=True, weights=sweights, focal=False, gamma=1)

        self.train_sdice_seg = costfunction.weightedDiceLoss( \
            self.y_seg, self.y_pseg, invarea=False, \
            weighting=True, weights=sweights2, hard=False, gamma=1)

        # Tumor dice only
        if self.flags.water_mask:
            hdice_w = [0.0,1.0,0.0]
        else:
            hdice_w = [0.0,1.0]
        self.train_hdice_seg = costfunction.weightedDiceLoss( \
            self.y_seg, self.y_pseg, invarea=False, \
            weighting=True, weights=hdice_w, hard=True, gamma=1)

        '''
        # Water mask
        if self.flags.water_mask:
            self.train_wdice_seg = costfunction.weightedDiceLoss( \
                self.y_seg, self.y_pseg, invarea=False, zero_as_one=False, \
                weighting=True, weights=[0.0,0.0,1.0], hard=True, gamma=1)
        else:
            self.train_wdice_seg = tf.constant(0, dtype=tf.float32)
        '''
        tf.summary.scalar('cls0_accuracy', self.train_accu_cls0)
        tf.summary.scalar('cls1_accuracy', self.train_accu_cls1)

        tf.summary.scalar('cls0_xentropy', self.train_xent_cls0)
        tf.summary.scalar('cls1_xentropy', self.train_xent_cls1)

        tf.summary.scalar('seg_accuracy', self.train_accu_seg)
        tf.summary.scalar('seg_xentropy', self.train_xent_seg)

        tf.summary.scalar('seg_sdice', self.train_sdice_seg)
        tf.summary.scalar('seg_hdice', self.train_hdice_seg)
        #tf.summary.scalar('seg_wdice', self.train_wdice_seg)

        # Mix training costs
        self.train_cost = tf.constant(0, dtype=tf.float32)
        if self.flags.train_seg:
            self.train_cost += self.flags.seg_loss_coefficient*( \
                self.train_sdice_seg + 100*self.train_xent_seg)
        if self.flags.train_cls:
            assert (self.flags.big_cls or self.flags.small_cls), \
                "ERROR: train_cls is TRUE but big_cls && small_cls are FALSE"
            cls_cost = tf.constant(0, dtype=tf.float32)
            if self.flags.big_cls:
                cls_cost += self.train_xent_cls0
            if self.flags.small_cls:
                cls_cost += self.train_xent_cls1
            self.train_cost += cls_cost*self.flags.cls_loss_coefficient
        if self.flags.l2_regularizer:
            self.train_cost += tf.losses.get_regularization_loss()
        tf.summary.scalar('train_cost', self.train_cost)

        # Minimizer vars
        with tf.name_scope("global_iterator"):
            self.global_step = \
                tf.Variable(0, trainable=False, name='gstep', dtype=tf.int32)
            self.global_epoch = \
                tf.Variable(0, trainable=False, name='gepoch', dtype=tf.int32)
            self.slrn = tf.Variable( \
                 0, trainable=False, name='slrn', dtype=tf.int32)
            self.dlrn = tf.Variable( \
                -1, trainable=False, name='dlrn', dtype=tf.int32)
            self.rlrn = tf.Variable( \
                 1, trainable=False, name='dlrn', dtype=tf.float32)
        self.increment_global_epoch = \
            tf.assign(self.global_epoch, self.global_epoch+1)
        self.increment_global_step = \
            tf.assign(self.global_step, self.global_step+1)
        self.flip_dlrn = tf.assign(self.dlrn, -1*self.dlrn)
        self.update_rlrn = tf.assign(self.rlrn, \
            self.flags.learning_range_decay_rate*self.rlrn)

        if self.batch_renorm:
            # Renorm increment
            # Batch renorm
            self.renorm_rmax_start, self.renorm_rmax_end, \
                self.rmax_increment_begin, self.rmax_increment_range = \
                self.flags.renorm_rmax
            self.renorm_dmax_start, self.renorm_dmax_end, \
                self.dmax_increment_begin, self.dmax_increment_range = \
                self.flags.renorm_dmax
            if hvd.rank() == 0:
                print("RENORM_RMAX   : ", (self.renorm_rmax_start, \
                    self.renorm_rmax_end, self.rmax_increment_begin, \
                    self.rmax_increment_range))
                print("RENORM_DMAX   : ", (self.renorm_dmax_start, \
                    self.renorm_dmax_end, self.dmax_increment_begin, \
                    self.dmax_increment_range))

            dv_rmax = self.renorm_rmax_end - self.renorm_rmax_start
            ds_rmax = tf.clip_by_value(tf.cast( \
                self.global_step - self.rmax_increment_begin, \
                dtype=tf.float32)/tf.cast(self.rmax_increment_range, \
                tf.float32), 0.0, 1.0)
            self.increment_rmax = tf.assign(self.renorm_rmax, \
                self.renorm_rmax_start + dv_rmax*ds_rmax)
            dv_dmax = self.renorm_dmax_end - self.renorm_dmax_start
            ds_dmax = tf.clip_by_value(tf.cast( \
                self.global_step - self.dmax_increment_begin, \
                dtype=tf.float32)/tf.cast(self.dmax_increment_range, \
                tf.float32), 0.0, 1.0)
            self.increment_dmax = tf.assign(self.renorm_dmax, \
                self.renorm_dmax_start + dv_dmax*ds_dmax)

        # Learning rates
        self.istep_max = \
            int(np.ceil(np.ceil(1.0*ntrain/hvd.size())/self.batchsize))
        self.learning_rate_stepsize = \
            self.flags.learning_rate_epochsize*self.istep_max
        lrn_range = self.flags.max_learning_rate - self.flags.min_learning_rate
        lrn_sfrac = tf.cast(self.slrn, tf.float32)/tf.cast( \
            self.learning_rate_stepsize, tf.float32)
        self.base_learning_rate = \
            self.flags.min_learning_rate + lrn_sfrac*lrn_range*self.rlrn
        # Scale lrn by # hvd nodes
        if self.flags.learning_rate_decay_step <=0 or \
            self.flags.learning_rate_decay_rate <=0:
            self.learning_rate = hvd.size()*self.base_learning_rate
        else:
            self.learning_rate = hvd.size()*tf.train.exponential_decay( \
                self.base_learning_rate, self.global_step, \
                self.flags.learning_rate_decay_step, \
                self.flags.learning_rate_decay_rate, staircase=True)
        self.increment_slrn = tf.assign(self.slrn, self.slrn + self.dlrn)

        tf.summary.scalar('global_step', self.global_step)
        tf.summary.scalar('global_epoch', self.global_epoch)
        tf.summary.scalar('learning_rate', self.learning_rate)

        seg_opt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
            os.path.join(self.model_scope,'model_input/'))
        seg_opt += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
            os.path.join(self.model_scope,'model_seg/'))
        cls_opt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
            os.path.join(self.model_scope,'model_cls/'))
        cls_opt += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, \
            os.path.join(self.model_scope,'model_fcn/'))
        if self.flags.train_seg and self.flags.train_cls:
            opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        elif self.flags.train_seg:
            opt_vars = seg_opt
        elif self.flags.train_cls:
            opt_vars = cls_opt
        else:
            opt_vars = []
        if hvd.rank() == 0:
            print("\nOPT SCOPE")
            for iopt in opt_vars:
                print(iopt)
        # Fix batch norm not updating running stdev and mean during training
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            opt = hvd.DistributedOptimizer( \
                optimizer_dict[self.flags.optimizer](self.learning_rate))
            opt_gradients, opt_variables = zip(*opt.compute_gradients( \
                self.train_cost, var_list=opt_vars))
            #opt_gradients, _ = tf.clip_by_global_norm(opt_gradients, 5.0)
            self.optimizer = opt.apply_gradients( \
                zip(opt_gradients, opt_variables))
        if hvd.rank() == 0:
            print("\nUPDATE OPS:")
            for ivar in self.update_ops:
                print(ivar)
            sys.stdout.flush()

        # Save/Restore
        if hvd.rank() == 0:
            self.global_iterator_vars = tf.get_collection( \
                tf.GraphKeys.GLOBAL_VARIABLES, scope='global_iterator')
            self.model_vars = []
            if self.flags.restore_seg:
                self.model_vars += self.model_input + self.model_seg
            if self.flags.restore_cls:
                self.model_vars += self.model_cls + self.model_fcn
            self.saver = tf.train.Saver(max_to_keep=None)
            if self.flags.restore:
                if self.flags.continuation:
                    self.restorer = tf.train.Saver()
                else:
                    self.restorer = tf.train.Saver(var_list = self.model_vars)

        if hvd.rank() == 0:
            print("\nAll variables:")
            for ivar in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(ivar)
            sys.stdout.flush()

        # Tensorboard writer
        self.tf_summary = tf.summary.merge_all()
        if log_path is not None:
            self.train_writer = \
                tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        else:
            self.train_writer = None

        if hvd.rank() == 0:
            print("Start session")
            sys.stdout.flush()

        # Dictionaries to keep track of output tensors for sess.run()
        self.train_accumulator = costfunction.costAccumulator( \
            collections.OrderedDict([ \
            ("train_cost"     , self.train_cost), \
            ("train_accu_cls0", self.train_accu_cls0), \
            ("train_accu_cls1", self.train_accu_cls1), \
            ("train_accu_seg" , self.train_accu_seg),  \
            ("train_xent_cls0", self.train_xent_cls0), \
            ("train_xent_cls1", self.train_xent_cls1), \
            ("train_xent_seg" , self.train_xent_seg),  \
            ("train_sdice_seg", self.train_sdice_seg), \
            ("train_hdice_seg", self.train_hdice_seg), \
            #("train_wdice_seg", self.train_wdice_seg), \
            ]))
        self.batch_accumulator = costfunction.costAccumulator( \
            collections.OrderedDict([ \
            ("train_cost"     , self.train_cost), \
            ("train_accu_cls0", self.train_accu_cls0), \
            ("train_accu_cls1", self.train_accu_cls1), \
            ("train_accu_seg" , self.train_accu_seg),  \
            ("train_xent_cls0", self.train_xent_cls0), \
            ("train_xent_cls1", self.train_xent_cls1), \
            ("train_xent_seg" , self.train_xent_seg),  \
            ("train_sdice_seg", self.train_sdice_seg), \
            ("train_hdice_seg", self.train_hdice_seg), \
            #("train_wdice_seg", self.train_wdice_seg), \
            ]))
        self.valid_accumulator = costfunction.costAccumulator( \
            collections.OrderedDict([ \
            ("valid_cost"     , self.train_cost), \
            ("valid_accu_cls0", self.train_accu_cls0), \
            ("valid_accu_cls1", self.train_accu_cls1), \
            ("valid_accu_seg" , self.train_accu_seg),  \
            ("valid_xent_cls0", self.train_xent_cls0), \
            ("valid_xent_cls1", self.train_xent_cls1), \
            ("valid_xent_seg" , self.train_xent_seg),  \
            ("valid_sdice_seg", self.train_sdice_seg), \
            ("valid_hdice_seg", self.train_hdice_seg), \
            #("valid_wdice_seg", self.train_wdice_seg), \
            ]))
        self.input_tensor_dict = { \
            't1_in'  : self.t1_x, \
            't2_in'  : self.t2_x, \
            'tc_in'  : self.tc_x, \
            'examno' : self.y_examno, \
        }
        self.output_tensor_dict = { \
            'cls0_act_onehot'  : self.y0_act, \
            'cls1_act_onehot'  : self.y1_act, \
            'cls0_prd_onehot'  : self.y0_pred, \
            'cls1_prd_onehot'  : self.y1_pred, \
            'cls0_prd_softmax' : self.y0_pred_sfmax, \
            'cls1_prd_softmax' : self.y1_pred_sfmax, \
            'seg_act'          : self.y_seg, \
            'seg_prd'          : self.y_pseg, \
            'channels_in'      : self.channels_x, \
        }
        self.ncon0 = np.zeros((self.nclass0,self.nclass0), dtype=int)
        self.ncon1 = np.zeros((self.nclass1,self.nclass1), dtype=int)
        self.ncon0_recv = np.zeros(self.ncon0.shape, dtype=int)
        self.ncon1_recv = np.zeros(self.ncon1.shape, dtype=int)
        self.feedDict = {self.training: True, \
            self.kprob: self.flags.keep_prob, \
            self.kprob_fcn_D1: self.flags.keep_prob_fcn_D1, \
            self.kprob_fcn_D2: self.flags.keep_prob_fcn_D2}
        self.best_validation = None

        # Initialize session
        self.sess = tf.Session(config = config_proto)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Broadcast initialized variables to all nodes
        hvd.broadcast_global_variables(0).run(session=self.sess)

        self.__training_initialized = True
        if hvd.rank() == 0:
            print("Initialized variables")
        MPI.COMM_WORLD.Barrier()

        if self.testing:
            print("TEST RUNMODEL INITIALIZATION RANK ", hvd.rank())

    # Train function
    def trainOneEpoch(self, iepoch, save=False, save_path=None, \
        save_every_nsteps=0, report_every_nsteps=0):
        '''
        Script to train 1 epoch on each call
        Called by trainModel() object instance's train() method
        '''
        assert report_every_nsteps >= 0, "Error: report_every must be >=0"
        assert save_every_nsteps >= 0, "Error: svae_every must be >=0"
        # Reinitialize iterator for each new epoch
        self.sess.run(self.train_iterator.initializer)

        self.gepoch = self.sess.run(self.increment_global_epoch)
        self.train_accumulator.resetCost()
        self.batch_accumulator.resetCost()
        istep = 0

        self.ncon0[:], self.ncon1[:] = (0,0)
        ncls, nprd, nchn = (None, None, None)
        # TEST
        sumblk = np.zeros((10,), float)
        sumvec = np.zeros((10,), float)
        sumepo = np.zeros((10,), float)
        sumbat = 0
        # Training batch loop
        while (istep < self.istep_max):
            self.sess.run(self.increment_global_step)
            # Update renorm parameters
            if self.batch_renorm:
                rmin_sb, rmax_sb, dmax_sb = self.sess.run([self.renorm_rmin, \
                    self.increment_rmax, self.increment_dmax])

            # Get np train data
            t1r, t2r, tcr, dwir, heatr, wheatr, lblr, examnor, ager, \
                sexr, locr = self.sess.run(self.train_next)
            feedDictt = self.feedDict
            feedDictt.update( \
                {self.t1_in: t1r, self.t2_in: t2r, self.tc_in: tcr, \
                self.dwi_in: dwir, self.heat_in: heatr, \
                self.wheat_in: wheatr, self.label_in: lblr, \
                self.examno_in: examnor, self.age_in: ager, \
                self.sex_in: sexr, self.loc_in: locr})

            if self.radiomics:
                # Get transformed data
                t1x, t2x = self.sess.run( \
                    [self.t1_x, self.t2_x], feed_dict=feedDictt)
                # Get radiomics from transformed data
                radiomicsx = radiomicsFeatures.getFeatures( \
                    np.concatenate([t1x, t2x], axis=-1))
                feedDictt.update({self.radiomics_x: radiomicsx})

            # Backpropagation & compute costs etc.
            _, cost_dict_sb, output_dict_sb, lrn_sb, gstep, \
                tf_summary_train = self.sess.run([self.optimizer, \
                self.batch_accumulator.getTensorDict(), \
                self.output_tensor_dict, self.learning_rate, \
                self.global_step, self.tf_summary], feed_dict=feedDictt)
            sb_size = output_dict_sb['cls1_act_onehot'].shape[0]

            # Gather costs
            self.batch_accumulator.setNewCost(cost_dict_sb, type='dict')
            self.train_accumulator.setNewCost(cost_dict_sb, type='dict')

            y0_pred_oh = costfunction.makeOneHot( \
                output_dict_sb['cls0_prd_softmax'])
            y1_pred_oh = costfunction.makeOneHot( \
                output_dict_sb['cls1_prd_softmax'])
            self.ncon0 += costfunction.confusionMatrix( \
                output_dict_sb['cls0_act_onehot'], y0_pred_oh, self.nclass0)
            self.ncon1 += costfunction.confusionMatrix( \
                output_dict_sb['cls1_act_onehot'], y1_pred_oh, self.nclass1)

            if ncls is None:
                ncls = np.sum( \
                    output_dict_sb['cls1_act_onehot'], axis=0).astype(int)
                nprd = np.sum(y1_pred_oh, axis=0).astype(int)
                nchn = np.sum( \
                    output_dict_sb['channels_in'], axis=0).astype(int)
            else:
                ncls += np.sum( \
                    output_dict_sb['cls1_act_onehot'], axis=0).astype(int)
                nprd += np.sum(y1_pred_oh, axis=0).astype(int)
                nchn += np.sum( \
                    output_dict_sb['channels_in'], axis=0).astype(int)
            nclsprd_send = np.asarray([ncls, nprd])
            nchn_send = nchn

            # Gather costs from each node
            self.batch_accumulator.MPIUpdateCostAndCount(sb_size)
            self.train_accumulator.MPIUpdateCostAndCount(sb_size)
            if self.train_writer is not None:
                self.train_writer.add_summary(tf_summary_train, gstep)

            # Save after certain number of steps
            if (self.flags.save and hvd.rank() == 0 and save_every_nsteps > 0 \
                and istep % save_every_nsteps == 0):
                save_path_return = self.saver.save(self.sess, save_path)
                print("Saved graph at step ", istep, " in " + save_path_return)
                sys.stdout.flush()

            if (report_every_nsteps > 0 and istep % report_every_nsteps == 0):
                # Gather act & pred class populations
                nclsprd_recv = np.zeros(nclsprd_send.shape, dtype=int)
                nchn_recv = np.zeros(nchn_send.shape, dtype=int)
                MPI.COMM_WORLD.Allreduce( \
                    nclsprd_send ,nclsprd_recv, op=MPI.SUM)
                MPI.COMM_WORLD.Allreduce(nchn_send ,nchn_recv, op=MPI.SUM)
                ncls, nprd = (nclsprd_recv[0,:], nclsprd_recv[1,:])
                nchn = nchn_recv
                # TEST
                nbatch = np.sum(ncls)
                if hvd.rank() == 0:
                    # Compute & print accumulated batch stats
                    print("Time: %s" % time.strftime("%c"))
                    # TEST
                    str_fmt = "Epoch: %5d gEpoch: %5d Step: %5d " + \
                        "gStep: %5d alpha: %8.6f %s"
                    print(str_fmt % (iepoch, self.gepoch, istep, gstep, \
                        lrn_sb, self.batch_accumulator.strCost(mean=True)))
                    if self.batch_renorm:
                        print("rmin: %7.4f  rmax: %7.4f  dmax: %7.4f" % \
                            (rmin_sb, rmax_sb, dmax_sb))
                    print("cls: %s\nprd: %s\nchn: %s" % (ncls, nprd, nchn))
                    sys.stdout.flush()
                self.batch_accumulator.resetCost()
                sys.stdout.flush()
                ncls, nprd, nchn = (None, None, None)

            # Update learning rates
            slrn, dlrn, blrn, rlrn = self.sess.run( \
                [self.slrn, self.dlrn, self.base_learning_rate, self.rlrn])
            if slrn >= self.learning_rate_stepsize and dlrn > 0:
                self.sess.run(self.flip_dlrn)
            elif slrn <= 0 and dlrn < 0:
                self.sess.run(self.flip_dlrn)
            self.sess.run(self.increment_slrn)
            istep += 1

        self.ncon0_recv[:] = 0
        self.ncon1_recv[:] = 0
        MPI.COMM_WORLD.Allreduce(self.ncon0, self.ncon0_recv, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(self.ncon1, self.ncon1_recv, op=MPI.SUM)
        if hvd.rank() == 0:
            print("TOTAL 0", np.sum(self.ncon0_recv))
            print("TOTAL 1", np.sum(self.ncon1_recv))
            print("Time: %s" % time.strftime("%c"))
            print("\tiepoch: %5d gEpoch: %5d %s" % (iepoch, self.gepoch, \
                self.train_accumulator.strCost(prefix='epoch_', mean=True)))
            print("\tconfusion_matrix 0:\n%s" % self.ncon0_recv)
            print("\tconfusion_matrix 1:\n%s" % self.ncon1_recv)
            sys.stdout.flush()

        # Update learning rate range after 1 cycle (2x rate epoch size)
        if self.gepoch%(2*self.flags.learning_rate_epochsize) == 0 \
            and self.flags.learning_range_decay:
            self.sess.run(self.update_rlrn)

        MPI.COMM_WORLD.Barrier()

        if self.testing:
            print("TEST TRAINONEEPOCH RANK ", hvd.rank())
            if hvd.rank() == 0:
                print("TEST TRAINONEEPOCH TOTAL %3d" % (np.sum(self.ncon0_recv)))
                print("TEST TRAINONEEPOCH CLS0", output_dict_sb['cls0_prd_onehot'].shape)
                print("TEST TRAINONEEPOCH SEGP", output_dict_sb['seg_prd'].shape)

    def validateOneEpoch(self, iepoch, save=False, save_path=None, \
        out_res_path=None, sel_threshold=[0,1.0], save_results=False):
        '''
        Script to validate once on each call
        - Validates model based on fixed validation set from data_object's pipeline
        - Saves classification & segmentation outputs of model into pckl objects
        Called by trainModel() object instance's train() method
        '''
        assert (sel_threshold[0] >= 0 and sel_threshold[1] <= 1.0 \
            and sel_threshold[1] > sel_threshold[0]), \
            "Error: invalid sel_threshold"
        # Validation
        self.valid_accumulator.resetCost()

        # Container for printout
        out_res = []
        # Validation batch loop
        self.sess.run(self.valid_iterator.initializer)

        self.ncon0[:] = 0
        self.ncon1[:] = 0
        vstep = 0
        mpi_all_done = False
        mpi_rank_done = False
        while (not mpi_all_done):
            # Fetch batch from tfdataset and shuffle
            try:
                t1r, t2r, tcr, dwir, heatr, wheatr, lblr, examnor, ager, \
                    sexr, locr = self.sess.run(self.valid_next)
            except tf.errors.OutOfRangeError:
                t1r, t2r, tcr, dwir, heatr, wheatr, lblr, examnor, ager, \
                    sexr, locr = self.data_object.dummyData()
                mpi_rank_done = True

            # Channel mask for validation
            t1r  = t1r*self.flags.series_val[0]
            t2r  = t2r*self.flags.series_val[1]
            tcr  = tcr*self.flags.series_val[2]
            dwir = dwir*self.flags.series_val[3]

            feedDictv = {self.t1_in: t1r, self.t2_in: t2r, self.tc_in: tcr, \
                self.dwi_in: dwir, self.heat_in: heatr, \
                self.wheat_in: wheatr, self.label_in: lblr, \
                self.examno_in: examnor, self.sex_in: sexr, \
                self.loc_in: locr, self.training:False, self.kprob: 1.0, \
                self.kprob_fcn_D1: 1.0, self.kprob_fcn_D2: 1.0}

            if self.radiomics:
                # Get transformed data
                t1x, t2x = self.sess.run([self.t1_x, self.t2_x], \
                    feed_dict=feedDictv)
                # Get radiomics from transformed data
                radiomicsx = radiomicsFeatures.getFeatures( \
                    np.concatenate([t1x, t2x], axis=-1))
                feedDictv.update({self.radiomics_x: radiomicsx})

            # Calculate costs & model outputs
            cost_dict_sb, output_dict_sb, input_dict_sb = \
                self.sess.run([self.valid_accumulator.getTensorDict(), \
                self.output_tensor_dict, self.input_tensor_dict], \
                feed_dict=feedDictv)

            self.valid_accumulator.setNewCost(cost_dict_sb, type='dict')
            if (not mpi_rank_done):
                sb_size = output_dict_sb['cls0_act_onehot'].shape[0]
            else:
                sb_size = 0

            # Gather cost from each node
            self.valid_accumulator.MPIUpdateCostAndCount(sb_size)
            if hvd.rank() == 0:
                if vstep%100 == 0:
                    print("Time: %s Step: %8d" % (time.strftime("%c"), vstep))
                    sys.stdout.flush()
            if not mpi_rank_done:
                y0_pred_oh = costfunction.makeOneHot( \
                    output_dict_sb['cls0_prd_softmax'])
                y1_pred_oh = costfunction.makeOneHot( \
                    output_dict_sb['cls1_prd_softmax'])
                self.ncon0 += costfunction.confusionMatrix( \
                    output_dict_sb['cls0_act_onehot'], \
                    y0_pred_oh, self.nclass0)
                self.ncon1 += costfunction.confusionMatrix( \
                    output_dict_sb['cls1_act_onehot'], \
                    y1_pred_oh, self.nclass1)
            vstep += 1

            # Accumulate data to be written out for each node
            if out_res_path is not None and save_results:
                if not mpi_rank_done:
                    isave_list = []
                    for isave in range(sb_size):
                        bg_ratio = output_dict_sb['seg_act'][isave,:,:,:,0]
                        bg_ratio = 1.0*np.sum(bg_ratio)/bg_ratio.size
                        if bg_ratio >= sel_threshold[0] \
                            and bg_ratio <= sel_threshold[1]:
                            isave_list.append(isave)
                    for isb in isave_list:
                        # Batch index from sess.run starts from 0
                        out_res.append((iepoch, isb, \
                            input_dict_sb['examno'][isb], \
                            input_dict_sb['t1_in'][isb], \
                            input_dict_sb['t2_in'][isb], \
                            input_dict_sb['tc_in'][isb], \
                            output_dict_sb['cls0_act_onehot'][isb], \
                            output_dict_sb['cls0_prd_softmax'][isb], \
                            output_dict_sb['cls1_act_onehot'][isb], \
                            output_dict_sb['cls1_prd_softmax'][isb], \
                            output_dict_sb['seg_act'][isb], \
                            output_dict_sb['seg_prd'][isb]))
            mpi_all_done = MPI.COMM_WORLD.allreduce(mpi_rank_done, op=MPI.LAND)

        self.ncon0_recv[:] = 0
        self.ncon1_recv[:] = 0
        MPI.COMM_WORLD.Allreduce(self.ncon0, self.ncon0_recv, op=MPI.SUM)
        MPI.COMM_WORLD.Allreduce(self.ncon1, self.ncon1_recv, op=MPI.SUM)
        if hvd.rank() == 0:
            print("TOTAL 0", np.sum(self.ncon0_recv))
            print("TOTAL 1", np.sum(self.ncon1_recv))
            print("Time: %s" % time.strftime("%c"))
            print("\tiEpoch: %5d gEpoch: %5d %s" % (iepoch, self.gepoch, \
                self.valid_accumulator.strCost(prefix='epoch_', mean=True)))
            print("\tconfusion_matrix 0:\n%s" % self.ncon0_recv)
            print("\tconfusion_matrix 1:\n%s" % self.ncon1_recv)
            sys.stdout.flush()

            # Best validation
            if self.best_validation is None:
                self.best_validation = \
                    self.valid_accumulator.getCost('valid_cost', mean=True)

            if self.valid_accumulator.getCost('valid_cost', mean=True) \
                < self.best_validation:
                bsuffix = (".best_g%04d_%s" % (self.gepoch, \
                    time.strftime('%Y-%m-%d-%H%M-%S')))
                self.best_validation = \
                    self.valid_accumulator.getCost('valid_cost', mean=True)
                print("New best validation cost: %10.5f" % \
                    self.best_validation)
            else:
                bsuffix = (".ckpt_g%04d_%s" % (self.gepoch, \
                    time.strftime('%Y-%m-%d-%H%M-%S')))

            if save and save_path is not None:
                save_path_return = \
                    self.saver.save(self.sess, save_path + bsuffix)
                print("Saved best validation graph in " + save_path_return)

        # Don't send to root
        # All processes write their own shard to separate files
        nskip = int(np.floor(1.0/self.flags.out_res_frac))
        outres_recv = [k for i,k in enumerate(out_res) \
            if i in range(0, len(out_res), nskip)]
        if out_res_path is not None and len(outres_recv) > 0:
            iout_start = 0
            iout_label = 0
            while iout_start < len(outres_recv):
                iout_end = min(iout_start + self.outres_size, len(outres_recv))
                try:
                    outname = out_res_path + "." + str(self.gepoch) + \
                        ".R" + str("%02d" % hvd.rank()) + \
                        "." + str("%02d" % iout_label)
                    print("Saving " + outname)
                    pickle.dump(outres_recv[iout_start:iout_end], \
                        open(outname, "wb"))
                except:
                    print("Unable to save out_res")
                sys.stdout.flush()
                iout_start += self.outres_size
                iout_label += 1
        MPI.COMM_WORLD.Barrier()

        if self.testing:
            print("TEST VALIDONEEPOCH RANK ", hvd.rank())
            if hvd.rank() == 0:
                print("TEST VALIDONEEPOCH TOTAL %3d" % (np.sum(self.ncon0_recv)))
                print("TEST VALIDONEEPOCH CLS0", output_dict_sb['cls0_prd_onehot'].shape)
                print("TEST VALIDONEEPOCH SEGP", output_dict_sb['seg_prd'].shape)

    def restoreModel(self, restore_path):
        '''
        Script to restore model when called
        Called by trainModel() instance's configureTraining() method
        '''
        if hvd.rank() == 0:
            if self.flags.restore:
                self.restorer.restore(self.sess, restore_path)
                print("Restored graph from " + restore_path)
        # Broadcast restored variables to all nodes
        hvd.broadcast_global_variables(0).run(session=self.sess)
        MPI.COMM_WORLD.Barrier()

    def saveModel(self, save_path):
        '''
        Script to save model when called
        Called by trainModel() instance's train() method
        '''
        # Save final step of the epoch
        if hvd.rank() == 0:
            save_path_return = self.saver.save(self.sess, save_path)
            print("Saved final graph in " + save_path_return)
        MPI.COMM_WORLD.Barrier()

    # Destroy session
    def finalize(self):
        '''
        Script to terminate TF session
        Called by trainModel() instance's finalize() method
        '''
        self.sess.close()
        MPI.COMM_WORLD.Barrier()
