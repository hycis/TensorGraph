#!/usr/bin/env python3
'''
#------------------------------------------------------------------------------
# Main script to read config file, set up & run training
# WLRRDDC
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

import argparse
import sys
import os
import traceback

# Horovod
import horovod.tensorflow as hvd
hvd.init()
if hvd.rank() == 0:
    print("WLRRDDC")
    print("Horovod initialized with %3d nodes" % hvd.size())

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir)

import nn.run.train as train
import nn.run.configReader as configReader

# MPI
import mpi4py.rc
mpi4py.rc.initialize = False # Do not initialize MPI
mpi4py.rc.finalize = False
import mpi4py.MPI as MPI

if __name__ == '__main__':
    '''
    Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<list of GPUs>
        mpirun -np <# CPU cores, 1 for each GPU> -bind-to none -map-by slot
        main_train.py <training config ini>
    '''
    try:
        # Print MPI environment
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("# MPI Processes: " + str(MPI.COMM_WORLD.Get_size()))

        # Parse arguments
        flags = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            # Parse arguments
            parser = argparse.ArgumentParser(description='Trains NN model')
            parser.add_argument('config', type=str, help='config file for training')
            parser.add_argument('--data_dir', type=str, help='abs path of data dir (overwrites INI)', default=None)
            args = parser.parse_args()

            # Read config file for training
            flags = configReader.parameters(args.config)
            if args.data_dir is not None:
                if flags.data_dir is not None:
                    print("WARNING: Overwriting data_dir with cmd argument: ", flags.data_dir)
                flags.data_dir = args.data_dir

            assert flags.model_module is not None, "ERROR: No model specified"
            assert flags.data_module is not None, "ERROR: No data specified"

            print("TRAINING CONFIGURATION")
            flags.listFlags()
            print("")
        flags = MPI.COMM_WORLD.bcast(flags, root=0)

        # Load model & set up training routines
        if flags.batch_renorm:
            batch_renorm_rmax_dmax = (flags.renorm_rmax[0], flags.renorm_dmax[0])
        else:
            batch_renorm_rmax_dmax = None
        train_model = train.trainModel(flags.data_module, flags.model_module, \
            flags.model_scope, flags.batchsize, flags.l2_weight_decay, \
            batch_renorm_rmax_dmax, flags.anatomies, flags.biopsy_only, \
            flags.mask_only, flags.water_mask, flags.series_kprob, \
            flags.train_valid_seed, flags.validation_list, flags.radiomics, \
            flags.clsmatcoeff, flags.distributed_batchnorm, \
            flags.data_dir, flags.testing)

        train_model.configureTraining(flags, \
            flags.save_path, flags.restore_path, flags.log_path)

        # Train model according to config file parameters
        train_model.train()

        # Finalize train_model object
        train_model.finalize()

        if flags.testing:
            print("TEST MAINTRAIN DONE RANK ", hvd.rank())
        # Force terminate after this point (evyerthing works, MPI just refuses to return)
        MPI.COMM_WORLD.Abort(0)

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        print("NODE ", hvd.rank(), ": ERROR running script")
        MPI.COMM_WORLD.Abort(1)
        sys.exit(1)
