'''
#------------------------------------------------------------------------------
# TF Dataset object for training & validation using TFGRAND5
#------------------------------------------------------------------------------
# Features:
#   Input/Output
#       - Input (from GRAND5)
#         Dimension : 24x320x320
#         Intensity range : 0 to approx. 255 (can exceed this)
#         Channels: t1, t2, tc, dwi b1000, tumor mask, water mask
#       - Ouptut
#         Dimension: 24x256x256 (resized, currently not cropped)
#         Intensity range : 0 to 2.0 (rescaled by dividing by 255)
#         Channels: t1, t2, tc, dwi (adc WIP), 3 channel tumor+water+bg mask
#
#   Data Augmentation
#       - Series deregistration & overall rotation
#       - Spatial flipUD, flipLR, x-y offset, x-y scaling
#       - Intensity DC offset, scaling
#       - Random series dropout
#       - Class over/undersampling (for training set only)
#
#   Heirarchical classification
#       - Automatic small & big tumor classes groupings
#       - Adjacency matrix for edges between small & big classes (NClassMat)
#         Linked classes receives a weight of 1, unrelated classes -0.5
#
#   Data Selection
#       - Based on Jiahao's TFGRAND5 file path selection script
#       - Random seed can be used to generate train/validation split. This
#         random seed will also affect the random placement of data onto
#         MPI nodes
#       - Alternatively, seperate fixed list (read from text file) of tfrecord
#         paths can be fed as validation set
#       - Overlaps between train & validation sets will be removed with
#         priority given to validation
#------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

import numpy as np
import glob
import fnmatch
import random
import sys
import os
import tensorflow as tf

# Do not initialize MPI
import mpi4py.rc
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import mpi4py.MPI as MPI

# Randomization functions
def randomAngle(deg=40, N=1):
    return tf.random_uniform([N], \
        minval=-deg/180.*np.pi, maxval=deg/180.*np.pi)
def randomSizes(xy_size):
    RAND = tf.random_uniform([2], minval=0.85, maxval=1.1, dtype=tf.float32)
    newsize = tf.convert_to_tensor(xy_size, dtype=tf.float32)*RAND
    return tf.cast(newsize, tf.int32)
def randomBool():
    RAND = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.int32)[0]
    return tf.cast(RAND, tf.bool)
def zoomTF(x, image_shape, size, tr_only=False):
    with tf.name_scope('zoomTF'):
        zsize = tf.cast(size, tf.float32)
        h_frac = 1.0*image_shape[1]/zsize[0]
        w_frac = 1.0*image_shape[2]/zsize[1]
        hd = 0.5*h_frac*(zsize[0] - image_shape[1])
        wd = 0.5*w_frac*(zsize[1] - image_shape[2])
        zoom_tr = tf.convert_to_tensor([h_frac, 0, hd, 0, w_frac, wd, 0, 0])
        zoom_tr = tf.expand_dims(zoom_tr, axis=0)
        if tr_only:
            out = zoom_tr
        else:
            out = tf.contrib.image.transform( \
                x, zoom_tr, interpolation='BILINEAR')
    return out
def rotateTF(x, image_shape, angle_rad, tr_only=False):
    # angle_rad can be list of angles
    with tf.name_scope('rotateTF'):
        rotate_tr = tf.contrib.image.angles_to_projective_transforms(\
            angle_rad, image_shape[1], image_shape[2])
        if tr_only:
            out = rotate_tr
        else:
            out = tf.contrib.image.transform( \
                x, rotate_tr, interpolation='BILINEAR')
    return out
def flipLR(x, image_shape, flip, tr_only=False):
    with tf.name_scope('randomFlip'):
        # vol must be of shape [batch or z, y, x, channels]
        flip_tr = tf.convert_to_tensor(
            [-1., 0., image_shape[2], 0., 1., 0., 0., 0.], dtype=tf.float32)
        flip_id = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if tr_only:
            out = tf.cond(flip, lambda: flip_tr, lambda: flip_id)
            out = tf.expand_dims(out, axis=0)
        else:
            out = tf.cond(flip, lambda: tf.contrib.image.transform( \
                x, flip_tr, interpolation='BILINEAR'), lambda: x)
    return out
def flipUD(x, image_shape, flip, tr_only=False):
    with tf.name_scope('randomFlip'):
        # vol must be of shape [batch or z, y, x, channels]
        flip_tr = tf.convert_to_tensor(
            [1., 0., 0., 0., -1., image_shape[1], 0., 0.], dtype=tf.float32)
        flip_id = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if tr_only:
            out = tf.cond(flip, lambda: flip_tr, lambda: flip_id)
            out = tf.expand_dims(out, axis=0)
        else:
            out = tf.cond(flip, lambda: tf.contrib.image.transform( \
                x, flip_tr, interpolation='BILINEAR'), lambda: x)
    return out
def randomOffset(xy_size):
    RAND = tf.random_uniform([2], minval=-0.1, maxval=0.1, dtype=tf.float32)
    offset = tf.convert_to_tensor(xy_size, dtype=tf.float32)*RAND
    return tf.cast(offset, tf.int32)
def offsetTF(x, image_shape, xy_offset, tr_only=False):
    with tf.name_scope('randomOffset'):
        # vol must be of shape [batch or z, y, x, channels]
        xy_offset = tf.cast(xy_offset, tf.float32)
        offset_tr = tf.convert_to_tensor( \
            [1., 0., xy_offset[0], 0., 1., xy_offset[1], 0., 0.], dtype=tf.float32)
        if tr_only:
            out = tf.expand_dims(offset_tr, axis=0)
        else:
            out = tf.contrib.image.transform(x, offset_tr, interpolation='BILINEAR')
    return out

def CenterBoxTF(data, box):
    def shapeDHW(data):
        return data.shape[0].value, data.shape[1].value, data.shape[2].value
    D,H,W = shapeDHW(data)
    d,h,w,c  = box
    d,h,w = D-d, H-h, W-w
    d,h,w = int(d/2), int(h/2), int(w/2)
    return d, d+box[0], h, h+box[1], w, w+box[2]

def BoundingBoxTF(data, box, training=True):
    with tf.name_scope('BoundingBoxTF'):
        # Centers BBox to tumor mask
        def shapeDHW(data):
            return data.shape[0].value, \
                data.shape[1].value, data.shape[2].value
        assert isinstance(data, tf.Tensor), "Data is not Tensor"
        D,H,W = shapeDHW(data)
        #box  = tf.constant(box,dtype=tf.int32, shape=[4])
        #d,h,w    = box[0],box[1],box[2]
        d,h,w,c  = box
        d,h,w,c  = d/2., h/2., w/2., c
        d1,h1,w1 = tf.floor(d),tf.floor(h),tf.floor(w)
        d2,h2,w2 = tf.ceil(d), tf.ceil(h), tf.ceil(w)
        d1,h1,w1 = tf.cast(d1, dtype=tf.int32),tf.cast( \
            h1, dtype=tf.int32),tf.cast(w1, dtype=tf.int32)
        d2,h2,w2 = tf.cast(d2, dtype=tf.int32),tf.cast( \
            h2, dtype=tf.int32),tf.cast(w2, dtype=tf.int32)

        coord    = tf.where(data > 0)
        centroid = tf.reduce_mean(coord, 0)
        centroid = tf.cast(centroid, tf.int32)
        ## De-Centered the Centroid
        xshift   = tf.random_uniform([1],-1, 2, dtype=tf.int32)[0]
        ## De-Centered the Centroid
        shift    = tf.random_uniform([3],-37,38,dtype=tf.int32)
        x,y,z    = centroid[0], centroid[1], centroid[2]
        if training:
            x,y,z = centroid[0] + xshift, \
                centroid[1] + shift[1], centroid[2] + shift[2]
        minX,minY,minZ = tf.subtract(x,d1),tf.subtract(y,h1),tf.subtract(z,w1)
        boundX   = tf.maximum(0,-minX)
        boundY   = tf.maximum(0,-minY)
        boundZ   = tf.maximum(0,-minZ)
        maxX,maxY,maxZ = x + d2 + boundX, y + h2 + boundY, z + w2 + boundZ
        minX,minY,minZ = tf.maximum(0,minX), \
            tf.maximum(0,minY), tf.maximum(0,minZ)
        boundX   = tf.maximum(0, maxX-D)
        boundY   = tf.maximum(0, maxY-H)
        boundZ   = tf.maximum(0, maxZ-W)
        minX,minY,minZ = minX - boundX, minY - boundY, minZ - boundZ

    return minX, maxX, minY, maxY, minZ, maxZ, centroid

def TFAugmentation(t1, t2, tc, dwi, mask, wmask, \
    image_shape, crop_shape, Training=True):
    # Concat series before augmentation to perform operations simultaneously
    data_block = tf.concat([t1,t2,tc,dwi,mask,wmask], axis=-1)
    N = tf.shape(data_block)[-1]
    # Perform Any Augmentation using TF operation
    if Training:
        # Move 'N' axis to dim 0 --> [N, y, x, z]
        data_block = tf.transpose(data_block, [3, 1, 2, 0])

        # De-registration
        microDeg = 0.01
        microAngles = tf.concat([randomAngle(microDeg, N=N-1), \
            tf.convert_to_tensor([0.0], dtype=tf.float32)], axis=0)
        #data_block = rotateTF(data_block, image_shape, microAngles)
        dereg_tr = rotateTF(None, image_shape, microAngles, tr_only=True)

        # Random rotations
        angle = randomAngle()
        #data_block = rotateTF(data_block, image_shape, angle)
        rotate_tr = rotateTF(None, image_shape, angle, tr_only=True)
        rotate_tr = tf.tile(rotate_tr, [N,1])

        # Random displacement
        displacement = randomOffset(image_shape[1:3])
        offset_tr = offsetTF(None, image_shape, displacement, tr_only=True)
        offset_tr = tf.tile(offset_tr, [N,1])

        # Random zooms
        rescale = randomSizes(image_shape[1:3])
        #data_block = zoomTF(data_block, image_shape, rescale)
        zoom_tr = zoomTF(None, image_shape, rescale, tr_only=True)
        zoom_tr = tf.tile(zoom_tr, [N,1])

        # Random flip
        flip_lr = flipLR(None, image_shape, randomBool(), tr_only=True)
        flip_lr = tf.tile(flip_lr, [N,1])
        flip_ud = flipUD(None, image_shape, randomBool(), tr_only=True)
        flip_ud = tf.tile(flip_ud, [N,1])

        # Perform all transformations
        all_tr = [dereg_tr, rotate_tr, offset_tr, zoom_tr, flip_lr, flip_ud]
        all_tr = tf.contrib.image.compose_transforms(*all_tr)
        data_block = tf.contrib.image.transform( \
            data_block, all_tr, interpolation='BILINEAR')

        # Swap 'N' axis back to dim 3 --> [z, y, x, N]
        data_block = tf.transpose(data_block, [3, 1, 2, 0])

    minZ, maxZ, minY, maxY, minX, maxX = \
        CenterBoxTF(data_block, box=crop_shape)
    data_block = data_block[minZ:maxZ,minY:maxY,minX:maxX,:]

    # Un-concat & crop z series back to original channels
    t1    = data_block[:,:,:,0:1]
    t2    = data_block[:,:,:,1:2]
    tc    = data_block[:,:,:,2:3]
    dwi   = data_block[:,:,:,3:4]
    mask  = data_block[:,:,:,4:5]
    wmask = data_block[:,:,:,5:]

    return t1 ,t2 ,tc, dwi, mask, wmask

def TFDownsample(t1, t2, tc, dwi, mask, wmask, downsample_shape, image_shape):
    # downsample_shape, image_shape args = (batch or z, y, x, channels)
    with tf.name_scope('TFDownsample'):
        # Concat series before aug to perform operations simultaneously
        data_block = tf.concat([t1,t2,tc,dwi,mask,wmask], axis=-1)
        # Downsamples x-y only
        new_shape = tf.cast(downsample_shape[1:3], tf.float32)
        minH = int(0.5*(image_shape[1] - downsample_shape[1]))
        minW = int(0.5*(image_shape[2] - downsample_shape[2]))

        h_frac = 1.0*image_shape[1]/new_shape[0]
        w_frac = 1.0*image_shape[2]/new_shape[1]
        hd = 0.5*h_frac*(new_shape[0] - image_shape[1])
        wd = 0.5*w_frac*(new_shape[1] - image_shape[2])
        zoom_tr = [h_frac, 0, hd, 0, w_frac, wd, 0, 0]
        data_block = tf.contrib.image.transform( \
            data_block, zoom_tr, interpolation='BILINEAR')
        data_block = data_block[:,minH:minH+downsample_shape[1], \
            minW:minW+downsample_shape[2],:]

        # Un-concat series back to original channels
        t1    = data_block[:,:,:,0:1]
        t2    = data_block[:,:,:,1:2]
        tc    = data_block[:,:,:,2:3]
        dwi   = data_block[:,:,:,3:4]
        mask  = data_block[:,:,:,4:5]
        wmask = data_block[:,:,:,5:]

    return t1 ,t2 ,tc, dwi, mask, wmask

def selectedDiseasePath(ROOT, disease_list, anatomy_list, \
    with_biopsy, with_mask, sorted=False, from_list=None):
    '''
    Selects file paths for training based on Jiahao's TFGRAND5 name convention
    for tumor disease type, biopsy & mask availability, and anatomies.
    Inputs: ROOT - folder containing all TFGRAND5 records. Will be globbed for
                   list of files
            disease_list - list of diseases to be gathered e.g. ['nml', 'jzl'].
                           If set to None, all diseases are selected
            anatomy_list - list of anatomies to be selected e.g. ['123']. If
                           set to None, all anatomies are selected
            with_biopsy - if T, select only files with biopsy ground truth
            with_mask - if T, select only files with masks
            sorted - if T, sorts globbed files (essential if fixed
                     randomization seed is used later on for train/valid split)
            from_list - if not None, filenames from this list will be used for
                        selection based on disease type, biopsy & mask
                        availability, and anatomies instead of glob from ROOT.
                        If this is used, ROOT can be set to None
    Outputs: list of file paths (string) that meet selection criteria
    '''
    # Glob ROOT only once
    if not hasattr(selectedDiseasePath, "globbed_paths"):
        selectedDiseasePath.globbed_paths = []
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Globbing files...", end='')
            sys.stdout.flush()
            selectedDiseasePath.globbed_paths = glob.glob(ROOT + '/*')
            print("Done")
        selectedDiseasePath.globbed_paths = \
            MPI.COMM_WORLD.bcast(selectedDiseasePath.globbed_paths, root=0)
    disease_paths = []

    # Whether to select from a predefined list or from glob of ROOT
    if from_list is not None:
        paths = from_list
    else:
        paths = selectedDiseasePath.globbed_paths

    if len(disease_list) == 0 and len(anatomy_list) == 0:
        # select all diseases
        disease_paths = paths

    elif len(disease_list) == 0 and len(anatomy_list) > 0:
        for anatomy in anatomy_list:
            disease_paths += [f for f in paths \
                if all(a in os.path.basename(f).split('_')[1] \
                for a in list(anatomy))]

    elif len(disease_list) > 0 and len(anatomy_list) == 0:
        for disease in disease_list:
            disease_paths += [f for f in paths \
                if fnmatch.fnmatch(os.path.basename(f), disease + '*')]

    elif len(disease_list) > 0 and len(anatomy_list) > 0:
        for disease in disease_list:
            for anatomy in anatomy_list:
                fset = [f for f in paths \
                    if fnmatch.fnmatch(os.path.basename(f), disease + '*')]
                disease_paths += [f for f in fset \
                    if all(a in os.path.basename(f).split('_')[1] \
                    for a in list(anatomy))]

    # Remove duplicates
    disease_paths = list(set(disease_paths))
    if with_biopsy:
        disease_paths = [p for p in disease_paths \
            if os.path.basename(p).split('_')[2] == '1']
    if with_mask:
        disease_paths = [p for p in disease_paths \
            if any(i == '1' for i in os.path.basename(p).split('_')[3:7])]

    # Sort gathered file paths
    if sorted:
        disease_paths.sort()
    return disease_paths

class Data:
    # train_list, valid_list, num_reps_train_golden, ...
    # ... num_reps_train_extra, with_biopsy_golden, with_mask_golden
    TFLibrary = { \
        'others'  : [[], [], 5  , 2  , True, True], \
        'bpy'     : [[], [], 1  , 1  , True, True], \
        'tsjl'    : [[], [], 1  , 1  , True, True], \
        'nml'     : [[], [], 1  , 0.5, True, True], \
        'jzl'     : [[], [], 1  , 0  , True, True], \
        'xxxbl'   : [[], [], 1  , 0.3, True, True], \
        'ctl'     : [[], [], 1  , 0.2, True, True], \
        'jsl'     : [[], [], 1  , 1  , True, True], \
        'smxbl'   : [[], [], 1  , 1  , True, True], \
        'lygl'    : [[], [], 1  , 0.5, True, True], \
        'sgml'    : [[], [], 1  , 1  , True, True], \
        'jzmxbl'  : [[], [], 1  , 1  , True, True], \
        'xgmxbl'  : [[], [], 1  , 1  , True, True], \
        'xgwpxbl' : [[], [], 2  , 2  , True, True], \
        'szxbl'   : [[], [], 3  , 1  , True, True], \
        'lbl'     : [[], [], 3  , 1  , True, True], \
        'jxbl'    : [[], [], 1  , 1  , True, True], \
        'pynz'    : [[], [], 4  , 1  , True, True], \
        'mlcrt'   : [[], [], 3  , 1  , True, True], \
        'nnz'     : [[], [], 2  , 2  , True, True], \
        'jjmql'   : [[], [], 5  , 5  , True, True], \
        'zyl'     : [[], [], 5  , 5  , True, True], \
        'jtl'     : [[], [], 2  , 2  , True, True], \
        'cyxnz'   : [[], [], 6  , 3  , True, True], \
        'klxbl'   : [[], [], 6  , 3  , True, True], \
        'nnc'     : [[], [], 1  , 1  , True, True], \
        'DNET'    : [[], [], 3  , 3  , True, True], \
        'sjql'    : [[], [], 1  , 1  , True, True], \
        'hssl'    : [[], [], 6  , 6  , True, True]}

    # key in TFLibrary, orig TFRecord index, modified index
    DISEASE_MAPPER = { \
        'others':                         ('others',   0,  0, ''), \
        'biao_pi_yang':                   ('bpy',      1,  1, ''), \
        'ting_shen_jing_liu':             ('tsjl',     2,  2, ''), \
        'nao_mo_liu':                     ('nml',      3,  3, ''), \
        'jiao_zhi_liu':                   ('jzl',      4,  4, ''), \
        'xing_xing_xi_bao_liu':           ('xxxbl',    5,  5, ''), \
        'chui_ti_liu':                    ('ctl',      6,  6, ''), \
        'ji_suo_liu':                     ('jsl',      7,  7, ''), \
        'sui_mu_xi_bao_liu':              ('smxbl',    8,  8, ''), \
        'lu_yan_guan_liu':                ('lygl',     9,  9, ''), \
        'shi_guan_mo_liu':                ('sgml',    10, 10, ''), \
        'jiao_zhi_mu_xi_bao_liu':         ('jzmxbl',  11, 11, ''), \
        'xue_guan_mu_xi_bao_liu':         ('xgmxbl',  12, 12, ''), \
        'xue_guan_wai_pi_xi_bao_liu':     ('xgwpxbl', 13, 13, ''), \
        'sheng_zhi_xi_bao_liu':           ('szxbl',   14, 14, ''), \
        'lin_ba_liu':                     ('lbl',     15, 15, ''), \
        'jie_xi_bao_liu':                 ('jxbl',    16, 16, ''), \
        'pi_yang_nang_zhong':             ('pynz',    17, 17, ''), \
        'mai_luo_cong_ru_tou_zhuang_liu': ('mlcrt',   18, 18, ''), \
        'nao_nong_zhong':                 ('nnz',     19,  0, 'others'), \
        'jing_jing_mai_qiu_liu':          ('jjmql',   20, 20, ''), \
        'zhuan_yi_liu':                   ('zyl',     21, 21, ''), \
        'ji_tai_liu':                     ('jtl',     23, 23, ''), \
        'chang_yuan_xing':                ('cyxnz',   24,  0, 'others'), \
        'ke_li_xi_bao':                   ('klxbl',   25,  0, 'others'), \
        'nao_nang_cong':                  ('nnc',     26,  0, 'others'), \
        'DNET':                           ('DNET',    27, 27, ''), \
        'shen_jing_qiao_liu':             ('sjql',    28, 28, ''), \
        'hei_se_su_liu':                  ('hssl',    29,  0, 'others'), \
        }
    TFLibrary_REMAP = None
    DISEASE_REMAPPER = None
    TF_TRAIN_RECORD_PATH = None
    TOTAL_TRAIN_DATA = None
    TF_VALID_RECORD_PATH = None
    TOTAL_VALID_DATA = None

    image_size = None
    crop_size = None
    valid_batch_size = None

    # Disease grouping
    disease_set = [ \
        ('jzl','jzmxbl','sgml','xxxbl'), \
        ('szxbl','jtl'), \
        ('tsjl','sjql'), \
        ]

    def __init__(self, root_dir=None, anatomies=[], biopsy_only=False, mask_only=False, \
        series_kprob=(1,1,1,1), train_valid_seed=1234, valid_list=None, \
        clsmatcoeff=(0,1.0), water_mask=False, testing=False):
        self.testing = testing
        self.ROOT = root_dir
        assert os.path.exists(self.ROOT), "ERROR: Data root dir " + self.ROOT + " does not exist"
        if self.testing:
            for k in self.TFLibrary.keys():
                self.TFLibrary[k][2], self.TFLibrary[k][3] = (1,1)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("DATASET SCRIPT 2C-WD", end="")
            if self.testing:
                print(" TEST MODE")
            else:
                print("")
            if valid_list is not None:
                print("NOTE: Reading validation list from " + valid_list)

        mapper = {}
        for ikey, iattribute in self.DISEASE_MAPPER.items():
            if iattribute[3] == '' or iattribute[3] is None:
                ivalue = iattribute[:3] + (iattribute[0],)
            else:
                ivalue = iattribute
            mapper.update({ikey:ivalue})
        self.DISEASE_MAPPER = mapper

        assert len(series_kprob) >= 4, "ERROR: len(series_kprob) < 4"
        if len(series_kprob) > 4:
            if hvd.rank() == 0:
                print("WARNING: Truncating series_kprob to len 4")
        self.series_kprob = np.asarray(series_kprob[:4], dtype=np.float32)

        # cls mat coeff
        self.neg_matcoeff = clsmatcoeff[0]
        self.pos_matcoeff = clsmatcoeff[1]

        self.water_mask = water_mask

        # set of unique modified indices
        unq_mod = set(sorted(k[2] for k in self.DISEASE_MAPPER.values()))
        # set of unique modified keys
        unq_modkey = set([k[3] for k in self.DISEASE_MAPPER.values()])
        # make sure bijective modkey:mod
        assert len(unq_mod) == len(unq_modkey), \
            "ERROR: mod:modkey not bijective"
        # Remap to 0-n_class
        # dict to map modifiied indices to sequential indices
        mod_to_remap = {mod:remap for remap, mod in enumerate(unq_mod)}
        # dict to map orig index to sequential indices
        orig_to_remap = {orig:mod_to_remap[mod] \
            for key, orig, mod, modkey in self.DISEASE_MAPPER.values()}
        # convert orig_to_remap dict to indexed list for TF
        remap = max([k for k,v in orig_to_remap.items()])
        self.remap = np.zeros(remap+1, dtype=np.int32)
        for key, value in orig_to_remap.items():
            self.remap[key] = value

        # Create threadsafe Random object & initialize seed
        self.random = random.Random(train_valid_seed)

        # Init training set by selecting all diseases present in TFlibrary
        # 'golden' set (biopsy & mask present) is used for validation
        for idisease in self.TFLibrary.keys():
            disease_train = []
            disease_valid = []
            # Get config, bounds & count attributes for this disease
            biopsy_flag = self.TFLibrary[idisease][4]
            mask_flag = self.TFLibrary[idisease][5]
            disease_paths = selectedDiseasePath( \
                self.ROOT, [idisease], anatomies, \
                with_biopsy=biopsy_flag, with_mask=mask_flag, sorted=True)
            n_data = len(disease_paths)
            n_train_reps = self.TFLibrary[idisease][2]
            if n_data < 2:
                "WARNING: n_data < 2 for " + idisease
                n_valid = 0
            else:
                if valid_list is None:
                    # Compute approx. 20-39 validation cases
                    if 20.0/n_data < (1.0/5):
                        n_valid = 20 + (n_data-1)%20 + 1
                    elif n_data > 20:
                        n_valid = (n_data-1)%20 + 1
                    else:
                        n_valid = max(1, int(1.0*n_data/3))

                    # Split data into train & val, oversample based on config
                    assert n_valid < n_data, \
                        "ERROR: n_valid >= # Data for " + idisease
                else:
                    n_valid = 0

                if MPI.COMM_WORLD.Get_rank() == 0:
                    self.random.shuffle(disease_paths)
                disease_paths = MPI.COMM_WORLD.bcast(disease_paths, root=0)
                if n_train_reps >= 1:
                    disease_train = \
                        disease_paths[0:n_data - n_valid]*n_train_reps
                elif n_train_reps > 0:
                    disease_train = disease_paths[0:n_data - n_valid]
                    n_trunc = int(n_train_reps*len(disease_train))
                    disease_train = disease_train[0:n_trunc]
                else:
                    disease_train = []

            if n_valid > 0:
                disease_valid = disease_paths[n_data - n_valid:]
            else:
                disease_valid = []
            self.TFLibrary[idisease][0] += disease_train
            self.TFLibrary[idisease][1] += disease_valid

        # Add back non-biopsy & non-mask as training data
        for idisease in self.TFLibrary.keys():
            disease_train = []
            # Get config, bounds & count attributes for this disease
            disease_paths = selectedDiseasePath( \
                self.ROOT, [idisease], anatomies, \
                with_biopsy=biopsy_only, with_mask=mask_only, sorted=True)
            # Add only samples that have not been added previously
            # Remove overlaps with origianl training set
            disease_paths = [k for k in disease_paths \
                if k not in set(self.TFLibrary[idisease][0])]
            # Remove overlap with validation set
            disease_paths = [k for k in disease_paths \
                if k not in set(self.TFLibrary[idisease][1])]
            n_data = len(disease_paths)
            # Replicate n times
            n_train_reps = self.TFLibrary[idisease][3]
            if n_train_reps >= 1:
                disease_train = disease_paths*n_train_reps
            elif n_train_reps > 0:
                n_trunc = int(n_train_reps*len(disease_paths))
                disease_train = disease_paths[0:n_trunc]
            else:
                disease_train = []

            #print "ADD ", idisease, n_data, len(self.TFLibrary[idisease][0])
            # Append to original training list
            self.TFLibrary[idisease][0] += disease_train

        # If we are to read validation list from a file
        if valid_list is not None:
            file_list = None
            if MPI.COMM_WORLD.Get_rank() == 0:
                fopen = open(valid_list, 'r')
                file_list = fopen.read().splitlines()
                fopen.close()

                for idisease in self.TFLibrary.keys():
                    disease_paths = selectedDiseasePath( \
                        self.ROOT, [idisease], anatomies, \
                        with_biopsy=biopsy_flag, with_mask=mask_flag, \
                        sorted=True, from_list=file_list)
                    self.TFLibrary[idisease][1] = disease_paths

            for idisease in self.TFLibrary.keys():
                self.TFLibrary[idisease][1] = \
                    MPI.COMM_WORLD.bcast(self.TFLibrary[idisease][1], root=0)

        # Make sure no overlap with validation
        for idisease in self.TFLibrary.keys():
            valid_basename = [os.path.basename(k) \
                for k in self.TFLibrary[idisease][1]]
            self.TFLibrary[idisease][0] = \
                [k for k in self.TFLibrary[idisease][0] \
                if os.path.basename(k) not in valid_basename]

        # Create grand list for training
        self.TF_TRAIN_RECORD_PATH = []
        self.TF_VALID_RECORD_PATH = []
        for idisease, iattribute in self.TFLibrary.items():
            self.TF_TRAIN_RECORD_PATH += iattribute[0]
            self.TF_VALID_RECORD_PATH += iattribute[1]
        self.TOTAL_TRAIN_DATA = len(self.TF_TRAIN_RECORD_PATH)
        self.TOTAL_VALID_DATA = len(self.TF_VALID_RECORD_PATH)

        # Check validity of file paths
        for i in self.TF_TRAIN_RECORD_PATH:
            assert os.path.exists(i), 'no such file {}'.format(i)
        for i in self.TF_VALID_RECORD_PATH:
            assert os.path.exists(i), 'no such file {}'.format(i)

        # Shuffle val data on rank 0 then bcast (may be redundant?)
        if MPI.COMM_WORLD.Get_size() > 1:
            if MPI.COMM_WORLD.Get_rank() == 0:
                self.random.shuffle(self.TF_VALID_RECORD_PATH)
            self.TF_VALID_RECORD_PATH = \
                MPI.COMM_WORLD.bcast(self.TF_VALID_RECORD_PATH, root=0)

        # Shuffle train data on rank 0 then bcast
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.random.shuffle(self.TF_TRAIN_RECORD_PATH)
        self.TF_TRAIN_RECORD_PATH = \
            MPI.COMM_WORLD.bcast(self.TF_TRAIN_RECORD_PATH, root=0)

        # cls weight distribution
        # dict to map from orig key to mod key
        key_to_modkey = {v[0]:v[3] for v in self.DISEASE_MAPPER.values()}
        # Convert TFLibrary to TFLibrary_REMAP
        self.TFLibrary_REMAP = {}
        for ikey, iattribute in self.TFLibrary.items():
            modkey = key_to_modkey[ikey]
            if modkey in self.TFLibrary_REMAP:
                self.TFLibrary_REMAP[modkey][0] += iattribute[0]
                self.TFLibrary_REMAP[modkey][1] += iattribute[1]
            else:
                self.TFLibrary_REMAP[modkey] = iattribute[:2]

        # convert DISEASE_MAPPER value tuple to dict
        # dmap = map of modkey to mod index
        # dict to map modkey to sequential indices  # {'bpy':1, ...}
        self.dmap = {key_to_modkey[v[0]]:orig_to_remap[v[1]] \
            for k, v in self.DISEASE_MAPPER.items()}
        # Ensure no indices are duplicated
        cls_lbl = [v for k, v in self.dmap.items()]
        assert len(set(cls_lbl)) == len(cls_lbl), "Error: Duplicate cls label"
        assert (max(cls_lbl)+1 == len(cls_lbl) and min(cls_lbl) == 0), \
            "Error: Class label not consecutive from 0 - # labels"
        cls_pop = np.zeros(len(cls_lbl), dtype=int)
        for imodkey, imod in self.dmap.items():
            n_train_data = len(self.TFLibrary_REMAP[imodkey][0])
            cls_pop[imod] += n_train_data
        cls_pop[cls_pop == 0] = -1
        cls_weights = 1.0/cls_pop
        cls_weights[cls_pop < 0] = 0
        cls_pop[cls_pop < 0] = 0
        cls_weights = cls_weights/max(np.max(cls_weights),1e-16)
        self.loss_clsweights1 = list(cls_weights)

        ## OTHER METADATA ##
        self.image_size = (24,320,320,1) # Orig size
        if self.testing:
            self.downsample_size = (24,128,128,1) # Downscale x-y
            self.crop_size = (24,128,128,1) # be careful if using unet maxpool
        else:
            self.downsample_size = (24,256,256,1) # Downscale x-y
            self.crop_size = (24,256,256,1) # be careful if using unet maxpool

        self.output_classes1 = len(cls_lbl)
        if self.water_mask:
            self.output_segclasses = 3 # BG=0/Tumor=1/Water=2
        else:
            self.output_segclasses = 2
        self.input_channels = 4 # [t1, t1c, t2, dwi]
        self.nloc = 4
        self.nage = 6 # None + >0,10,20,40,60
        self.nsex = 3 # None + M,F

        self.batchset = { \
            'train' : (self.TF_TRAIN_RECORD_PATH, self.TOTAL_TRAIN_DATA), \
            'valid' : (self.TF_VALID_RECORD_PATH, self.TOTAL_VALID_DATA)}
        self.mask_size = self.crop_size[0:-1] + (self.output_segclasses,)
        #segw = 1.0/np.array([11731282776, 52431861, 13562310])
        #segw = segw/np.sum(segw) [0.00091765 0.20531911 0.79376324]
        self.loss_segweights = [0.001, 0.3, 1.0]
        if not self.water_mask:
            self.loss_segweights = [0.001, 1.3]

        # Wasserstein
        reldisease = {}
        for iset in self.disease_set:
            for idisease in iset:
                reldisease[idisease] = iset
        self.tree_M = np.ones( \
            (self.output_classes1, self.output_classes1), dtype=np.float32)
        for idisease in self.dmap.keys():
            if idisease not in reldisease.keys():
                id = self.dmap[idisease]
                self.tree_M[id,id] = 0
            else:
                for ireldisease in reldisease[idisease]:
                    id = self.dmap[idisease]
                    ir = self.dmap[ireldisease]
                    if id == ir:
                        self.tree_M[id,ir] = 0
                    else:
                        self.tree_M[id,ir] = 0.5

        # Big grouping
        # Fine group index to parent fine group index
        self.fmap_to_lmap = {k:k for k in range(self.output_classes1)}
        for iset in self.disease_set:
            lid = self.dmap[iset[0]]
            for idisease in iset[1:]:
                fid = self.dmap[idisease]
                self.fmap_to_lmap[fid] = lid
        bmap = set(k for k in self.fmap_to_lmap.values())
        self.output_classes0 = len(bmap)
        # Parent fine group index to big group index
        self.lmap_to_bmap = {v:k for k,v in enumerate(sorted(bmap))}
        # Fine group index to big group index
        self.fmap_to_bmap = np.zeros(len(self.fmap_to_lmap), dtype=np.int32)
        for fmap, lmap in self.fmap_to_lmap.items():
            self.fmap_to_bmap[fmap] = self.lmap_to_bmap[lmap]
        #self.fmap_to_bmap = \
        #    {k:self.lmap_to_bmap[v] for k,v in self.fmap_to_lmap.items()}
        self.nclass_mat = self.neg_matcoeff*np.ones( \
            (self.output_classes0, self.output_classes1), dtype=np.float32)
        for fid in range(self.output_classes1):
            self.nclass_mat[self.fmap_to_bmap[fid],fid] = self.pos_matcoeff
        self.nclass_mat = np.transpose(self.nclass_mat)
        # Big class loss weights
        self.loss_clsweights0 = np.zeros( \
            self.output_classes0 ,dtype=np.float32)
        for fid in range(self.output_classes1):
            self.loss_clsweights0[self.fmap_to_bmap[fid]] += \
                1.0/self.loss_clsweights1[fid]
        self.loss_clsweights0 = 1.0/self.loss_clsweights0
        self.loss_clsweights0 /= np.max(self.loss_clsweights0)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("DATASET DIR : ", self.ROOT)
            print("ANATOMIES   : ", anatomies)
            print("BIOPSY ONLY : ", biopsy_only)
            print("MASK ONLY   : ", mask_only)
            print("CROP SIZE   : ", self.crop_size)
            print("NCLASS0      : ", self.output_classes0)
            print("NCLASS1      : ", self.output_classes1)
            np.set_printoptions(linewidth=256)
            print("NCLASSMAT    :\n%s" % np.transpose(self.nclass_mat))

    def listDataFiles(self, batchname):
        print("DATASET_NAME: ", batchname)
        for i, ifile in enumerate(self.batchset[batchname][0]):
            print("%5d %s" % (i, ifile))

    def getDataCount(self):
        sorted_info = [(self.dmap[k], k, len(v[0]), len(v[1]), \
            self.loss_clsweights1[self.dmap[k]]) \
            for k,v in self.TFLibrary_REMAP.items()]
        sorted_info = sorted(sorted_info, key=lambda x: x[0])
        print("%-2s %-10s %-5s %-5s %-7s" % \
            ("#", "cls_name", "n_trn", "n_val", "cls_wt"))
        for c_label, c_name, n_train, n_valid, c_weight in sorted_info:
            print("%2d %-10s %5d %5d %7.5f" % \
            (c_label, c_name, n_train, n_valid, c_weight))

        bmap_to_fmap = [[] for k in range(self.output_classes0)]
        for fid in range(self.output_classes1):
            bmap_to_fmap[self.fmap_to_bmap[fid]] += [fid]
        print("\nBig Classification")
        print("%-2s %-10s %-7s %-2s %-10s" % \
            ("#", "bcls_name", "bcls_wt", "#", "fcls_name"))
        for bid, fids in enumerate(bmap_to_fmap):
            first_fid = True
            for fid in fids:
                if first_fid:
                    b_name = sorted_info[fid][1]
                    b_wt = ("%7.5f" % self.loss_clsweights0[bid])
                    c_name = b_name
                    first_fid = False
                else:
                    bid = ""
                    b_name= ""
                    b_wt = ""
                    c_name = sorted_info[fid][1]
                print("%2s %-10s %7s %2s %-10s" % \
                    (str(bid), b_name, b_wt, fid, c_name))

    def getDataSize(self, batchname):
        return self.batchset[batchname][1]
    def getNLoc(self):
        return self.nloc
    def getNChannels(self):
        return self.input_channels
    def getOrigSize(self):
        return self.image_size
    def getCropSize(self):
        return self.crop_size
    def getMaskSize(self):
        return self.mask_size
    def getInputChannels(self):
        return self.input_channels
    def getOutputClasses0(self):
        return self.output_classes0
    def getOutputClasses1(self):
        return self.output_classes1
    def getOutputSegClasses(self):
        return self.output_segclasses
    def getValidBatchSize(self):
        return self.valid_batch_size
    def getTrainBatchSize(self):
        return self.train_batch_size
    def getTestBatchSize(self):
        return self.test_batch_size
    def getLossClsWeights1(self):
        return self.loss_clsweights1
    def getLossClsWeights0(self):
        return self.loss_clsweights0
    def getLossSegWeights(self):
        return self.loss_segweights
    def getNClassMat(self):
        return self.nclass_mat
    def getSize(self, batchname):
        return self.batchset[batchname][1]

    def readDecode(self, dataset_in):
        # Read raw TFRecord string and decode them into tensor etc. objects
        # Parse TFRecord entries
        feature_list = { \
            'age'       : tf.FixedLenFeature([], tf.int64),
            'gender'    : tf.FixedLenFeature([], tf.int64),
            'label'     : tf.FixedLenFeature([], tf.int64),
            'tail'      : tf.FixedLenFeature([], tf.int64),
            'cyst'      : tf.FixedLenFeature([], tf.int64),
            'examNo'    : tf.FixedLenFeature([], tf.int64),
            'anatomy'   : tf.FixedLenFeature([], tf.string),
            'mask_t2'   : tf.FixedLenFeature([], tf.string),
            'mask_tc'   : tf.FixedLenFeature([], tf.string),
            'mask_sagC' : tf.FixedLenFeature([], tf.string),
            'mask_corC' : tf.FixedLenFeature([], tf.string),
            't1'        : tf.FixedLenFeature([], tf.string),
            't2'        : tf.FixedLenFeature([], tf.string),
            'tc'        : tf.FixedLenFeature([], tf.string),
            'sagC'      : tf.FixedLenFeature([], tf.string),
            'corC'      : tf.FixedLenFeature([], tf.string),
            'sagDim'    : tf.FixedLenFeature([], tf.string),
            'corDim'    : tf.FixedLenFeature([], tf.string),
            'filename'  : tf.FixedLenFeature([], tf.string),
            'adc'       : tf.FixedLenFeature([], tf.string),
            'dwi1000'   : tf.FixedLenFeature([], tf.string)}
        features = tf.parse_single_example(dataset_in, features=feature_list)

        examno = tf.cast(features['examNo'], tf.int32)
        age    = tf.cast(features['age'], tf.int32)
        sex    = tf.cast(features['gender'], tf.int32)

        label = tf.cast(features['label'], tf.int32) # Classification index
        loc = tf.cast(tf.decode_raw(features['anatomy'], tf.float64), tf.int32)

        t1  = tf.decode_raw(features['t1'], tf.int16)
        t2  = tf.decode_raw(features['t2'], tf.int16)
        tc  = tf.decode_raw(features['tc'], tf.int16)
        dwi = tf.decode_raw(features['dwi1000'], tf.int16)
        mask = tf.decode_raw(features['mask_t2'], tf.int16)
        maskC = tf.decode_raw(features['mask_tc'], tf.int16)

        loc   = tf.reshape(loc, (self.nloc,), name="LOC_RESHAPE")
        t1    = tf.reshape(t1, self.image_size, name="T1_RESHAPE")
        t2    = tf.reshape(t2, self.image_size, name="T2_RESHAPE")
        tc    = tf.reshape(tc, self.image_size, name="TC_RESHAPE")
        dwi   = tf.reshape(dwi, self.image_size, name="DWI_RESHAPE")
        mask  = tf.reshape(mask, self.image_size, name="MASK_RESHAPE")
        maskC = tf.reshape(maskC, self.image_size, name="MASKC_RESHAPE")

        # Get tumor & water masks
        zeromask = tf.zeros(self.image_size, dtype=tf.int16)
        onesmask = tf.ones(self.image_size, dtype=tf.int16)
        # tumor mask (Cast water to 0)
        tmask  = tf.where(tf.equal(mask, 35), zeromask, mask)
        tmaskC = tf.where(tf.equal(maskC, 35), zeromask, maskC)
        # water mask
        wmask  = tf.where(tf.equal(mask, 35), onesmask, zeromask)
        wmaskC = tf.where(tf.equal(maskC, 35), onesmask, zeromask)

        t1    = tf.cast(t1, tf.float32)
        t2    = tf.cast(t2, tf.float32)
        tc    = tf.cast(tc, tf.float32)
        dwi   = tf.cast(dwi, tf.float32)
        mask  = tf.cast(mask, tf.float32)
        maskC = tf.cast(maskC, tf.float32)
        wmask = tf.cast(wmask, tf.float32)
        wmaskC = tf.cast(wmaskC, tf.float32)

        """ Make use of any Mask from T2 or TC """
        # Merge T1C and T2 mask
        heat = tf.round(tf.add(tmask, tmaskC))
        heat = tf.cast(tf.cast(heat, tf.bool), tf.int32)
        # Merge T1C and T2 mask
        wheat = tf.round(tf.add(wmask, wmaskC))
        wheat = tf.cast(tf.cast(wheat, tf.bool), tf.int32)

        # Remap label from 0-n_class
        remap = tf.convert_to_tensor(self.remap, dtype=tf.int32)
        label = remap[label]

        return t1, t2 ,tc, dwi, heat, wheat, label, examno, age, sex, loc

    def dataAugPrep(self, t1, t2, tc, dwi, heat, wheat, \
        label, age, sex, training=True):
        def age_bin(age):
            return tf.cond( \
                age <  0, lambda: 0, lambda: tf.cond( \
                age < 10, lambda: 1, lambda: tf.cond( \
                age < 20, lambda: 2, lambda: tf.cond( \
                age < 40, lambda: 3, lambda: tf.cond( \
                age < 60, lambda: 4, lambda: 5)))))

        def dropSeries(data, kprob, training):
            DROP = tf.cond(tf.greater(tf.random_uniform( \
                [1], minval=0, maxval=1, dtype=tf.float32)[0], kprob), \
                lambda: tf.constant(0, tf.float32), \
                lambda: tf.constant(1.0, tf.float32))
            data *= DROP
            return data

        def addNoiseNormalize(data, box, training):
            data = tf.divide(data, 255.)
            if training:
                # Random DC offset
                data += tf.random_normal([1], mean=0, stddev=0.06)[0]
                # Random brightness scaling
                data *= (1.0 + tf.random_normal([1], mean=0, stddev=0.06)[0])
                # Random noise
                data += tf.random_normal(box, mean=0, stddev=0.07)
            data = tf.clip_by_value(data, 0.0, 2.0)
            return data

        def getInChannels(t1, t2, tc, dwi):
            t1_in = tf.cast(tf.greater(tf.reduce_mean(t1), 1.1), tf.int32)
            t2_in = tf.cast(tf.greater(tf.reduce_mean(t2), 1.1), tf.int32)
            tc_in = tf.cast(tf.greater(tf.reduce_mean(tc), 1.1), tf.int32)
            dwi_in = tf.cast(tf.greater(tf.reduce_mean(dwi), 1.1), tf.int32)

            return tf.stack([t1_in, t2_in, tc_in, dwi_in], axis=0)

        # TEMP convert heat to float32 for rotation etc.
        heat  = tf.cast(heat, tf.float32)
        wheat = tf.cast(wheat, tf.float32)
        # Downsample
        t1, t2, tc, dwi, heat, wheat = TFDownsample( \
            t1, t2, tc, dwi, heat, wheat, self.downsample_size, self.image_size)
        # Distort input
        t1, t2, tc, dwi, heat, wheat = \
            TFAugmentation(t1, t2, tc, dwi, heat, wheat, \
            image_shape=self.downsample_size, crop_shape=self.crop_size, \
            Training=training)

        # Drop series
        t1  = dropSeries(t1 , self.series_kprob[0], training)
        t2  = dropSeries(t2 , self.series_kprob[1], training)
        tc  = dropSeries(tc , self.series_kprob[2], training)
        dwi = dropSeries(dwi, self.series_kprob[3], training)

        # Get InChannels
        channels_in = getInChannels(t1, t2, tc, dwi)

        # Add noise to records
        t1  = addNoiseNormalize(t1 , self.crop_size, training)
        t2  = addNoiseNormalize(t2 , self.crop_size, training)
        tc  = addNoiseNormalize(tc , self.crop_size, training)
        dwi = addNoiseNormalize(dwi, self.crop_size, training)

        # Convert segmentation mask to [0,1]
        heat = tf.round(heat)
        heat = tf.cast(tf.cast(heat, tf.bool), tf.int32)
        # Convert segmentation mask to [0,1]
        wheat = tf.round(wheat)
        wheat = tf.cast(tf.cast(wheat, tf.bool), tf.int32)
        # Convert classification label to one-hot with '0' as bg mask
        label1_onehot = tf.one_hot(label, self.output_classes1, dtype=tf.int32)
        # Remap label from 0-n_class
        remap = tf.convert_to_tensor(self.fmap_to_bmap, dtype=tf.int32)
        label0_onehot = tf.one_hot( \
            remap[label], self.output_classes0, dtype=tf.int32)

        # BG (no tumor & no water)
        heat_bg = tf.cast(tf.cast(tf.add(heat, wheat), tf.bool), tf.int32)
        heat_bg = tf.subtract(1, heat_bg)
        if self.water_mask:
            # Remove overlap between tumor & water by prioritizing tumor
            heat_wt = tf.clip_by_value(tf.subtract(tf.add(heat, wheat), 1), 0, 1)
            heat_wt = tf.clip_by_value(tf.subtract(wheat, heat_wt), 0, 1)
            # Concat mutually-exclusive masks as channels
            heat_seg = tf.concat([heat_bg, heat, heat_wt], axis=-1)
        else:
            heat_seg = tf.concat([heat_bg, 1-heat_bg], axis=-1)
        age_onehot = tf.one_hot(age_bin(age), self.nage, dtype=tf.int32)
        sex_onehot = tf.one_hot(sex, self.nsex, dtype=tf.int32)

        return t1, t2 ,tc, dwi, heat_seg, label0_onehot, label1_onehot, \
            age_onehot, sex_onehot, channels_in

    def batchAugPrep(self, t1, t2, tc, dwi, heat, wheat, label, \
        age, sex, tf_training):
        def mapWrapper(debatched_list, tf_training):
            t1    = debatched_list[0]
            t2    = debatched_list[1]
            tc    = debatched_list[2]
            dwi   = debatched_list[3]
            heat  = debatched_list[4]
            wheat = debatched_list[5]
            label = debatched_list[6]
            age   = debatched_list[7]
            sex   = debatched_list[8]

            out = tf.cond(tf_training, \
                lambda: self.dataAugPrep( \
                t1, t2, tc, dwi, heat, wheat, label, age, sex, training=True), \
                lambda: self.dataAugPrep( \
                t1, t2, tc, dwi, heat, wheat, label, age, sex, training=False))
            return out

        batch_tuple = (t1, t2, tc, dwi, heat, wheat, label, age, sex)
        batch_dtype = (tf.float32, tf.float32, tf.float32, tf.float32, \
            tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
        t1, t2, tc, dwi, heat_seg, label0_onehot, label1_onehot, \
            age_onehot, sex_onehot, channels_in = tf.map_fn( \
            lambda x: mapWrapper(x, tf_training), batch_tuple, \
            dtype=batch_dtype, back_prop=False)

        # Fix shape
        t1 = tf.reshape(t1, (-1,) + self.crop_size, name="PREP_T1_RESHAPE")
        t2 = tf.reshape(t2, (-1,) + self.crop_size, name="PREP_T2_RESHAPE")
        tc = tf.reshape(tc, (-1,) + self.crop_size, name="PREP_TC_RESHAPE")
        dwi = tf.reshape(dwi, (-1,) + self.crop_size, name="PREP_DWI_RESHAPE")
        heat_seg = tf.reshape(heat_seg, (-1,) + self.mask_size, \
            name="PREP_HEAT_RESHAPE")
        label0_onehot = tf.reshape(label0_onehot, (-1, self.output_classes0), \
            name="PREP_LABEL0_RESHAPE")
        label1_onehot = tf.reshape(label1_onehot, (-1, self.output_classes1), \
            name="PREP_LABEL1_RESHAPE")
        age_onehot = tf.reshape(age_onehot, \
            (-1,self.nage), name="PREP_NAGE_RESHAPE")
        sex_onehot = tf.reshape(sex_onehot, \
            (-1,self.nsex), name="PREP_NSEX_RESHAPE")
        channels_in = tf.reshape(channels_in, (-1,self.input_channels), \
            name="PREP_INCHANNELS_RESHAPE")
        return t1, t2, tc, dwi, heat_seg, label0_onehot, label1_onehot, \
            age_onehot, sex_onehot, channels_in

    def dummyData(self, n=1):
        dummy_t = np.zeros((n,) + self.image_size, dtype=np.float32)
        dummy_l = np.zeros((n,), dtype=np.int32)
        dummy_v = np.zeros((n,self.nloc), dtype=np.int32)
        return dummy_t, dummy_t, dummy_t, dummy_t, dummy_t, dummy_t, \
            dummy_l, dummy_l, dummy_l, dummy_l, dummy_v

    def reshuffleFileList(self, setname):
        file_list, num_records = self.batchset[setname]
        # Reshuffle root 0 then bcast
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.random.shuffle(file_list)
        file_list = MPI.COMM_WORLD.bcast(file_list, root=0)
        self.batchset[setname] = (file_list, num_records)

    def generateBatch(self, setname, batchsize, shufflesize, \
        shuffle_batch=True, num_shards=1, worker_rank=0, \
        repeat=-1, prefetch_gpu=False):
        assert (setname in self.batchset), "setname not in batchset"
        files = self.batchset[setname][0]

        with tf.device('/cpu:0'):
            dataset = tf.data.TFRecordDataset(files)
            dataset = dataset.shard(num_shards, worker_rank)

            if shuffle_batch:
                dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat( \
                    shufflesize, repeat, seed=1234))
            else:
                dataset = dataset.repeat(repeat)

            dataset = dataset.apply( \
                tf.contrib.data.map_and_batch(lambda x: self.readDecode(x), \
                batchsize, num_parallel_batches=6))

            if prefetch_gpu:
                dataset = dataset.apply(tf.contrib.data.prefetch_to_device( \
                    '/gpu:0', buffer_size=1))
            else:
                dataset = dataset.prefetch(buffer_size=1)

            iterator = dataset.make_initializable_iterator()

        return iterator
