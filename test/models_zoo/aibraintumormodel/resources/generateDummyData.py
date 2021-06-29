# Python2 compatibility
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import argparse

def generate_dummy_data(path, nperclass, classes=None, seed=None):
    def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    DISEASE_MAPPER = { \
         0: 'others', \
         1: 'bpy',    \
         2: 'tsjl',   \
         3: 'nml',    \
         4: 'jzl',    \
         5: 'xxxbl',  \
         6: 'ctl',    \
         7: 'jsl',    \
         8: 'smxbl',  \
         9: 'lygl',   \
        10: 'sgml',   \
        11: 'jzmxbl', \
        12: 'xgmxbl', \
        13: 'xgwpxbl',\
        14: 'szxbl',  \
        15: 'lbl',    \
        16: 'jxbl',   \
        17: 'pynz',   \
        18: 'mlcrt',  \
        19: 'nnz',    \
        20: 'jjmql',  \
        21: 'zyl',    \
        23: 'jtl',    \
        24: 'cyxnz',  \
        25: 'klxbl',  \
        26: 'nnc',    \
        27: 'DNET',   \
        28: 'sjql',   \
        29: 'hssl',   \
    }
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(1111)

    if classes is not None:
        for icls in classes:
            assert icls in DISEASE_MAPPER.values(), "ERROR: Invalid class given " + icls
        DISEASE_MAPPER = {k:v for k,v in DISEASE_MAPPER.items() if v in classes}
        print("Generating only for these classes: ", classes)

    ifile = 0
    sagcordim_str = np.array([12,440,440], dtype=np.int16).tostring()
    for label, disease in DISEASE_MAPPER.items():
        for isample in range(nperclass):
            ifile += 1
            filename = disease + "_1234_1_1_1_1_1_" + str(1000000+ifile) + ".tfrecords"
            filename = os.path.normpath(os.path.join(path, filename))
            feature_list = { \
                'age'      : _int64_feature(np.random.randint(0,100)), \
                'gender'   : _int64_feature(np.random.randint(0,3)), \
                'label'    : _int64_feature(label), \
                'tail'     : _int64_feature(np.random.randint(0,2)), \
                'cyst'     : _int64_feature(np.random.randint(0,2)), \
                'examNo'   : _int64_feature((1000000+ifile)), \
                'anatomy'   : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 2, size=(4,), dtype=np.int64).tostring())), \
                'mask_t2'   : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 2, size=(24,320,320,1), dtype=np.int16).tostring())), \
                'mask_tc'   : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 2, size=(24,320,320,1), dtype=np.int16).tostring())), \
                'mask_sagC' : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 2, size=(12,440,440,1), dtype=np.int16).tostring())), \
                'mask_corC' : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 2, size=(12,440,440,1), dtype=np.int16).tostring())), \
                't1'       : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(24,320,320,1), dtype=np.int16).tostring())), \
                't2'       : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(24,320,320,1), dtype=np.int16).tostring())), \
                'tc'       : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(24,320,320,1), dtype=np.int16).tostring())), \
                'sagC'     : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(12,440,440,1), dtype=np.int16).tostring())), \
                'corC'     : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(12,440,440,1), dtype=np.int16).tostring())), \
                'sagDim'   : _bytes_feature(tf.compat.as_bytes(sagcordim_str)), \
                'corDim'   : _bytes_feature(tf.compat.as_bytes(sagcordim_str)), \
                'filename' : _bytes_feature(filename.encode('utf-8')), \
                'adc'      : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(24,320,320,1), dtype=np.int16).tostring())), \
                'dwi1000'  : _bytes_feature(tf.compat.as_bytes(np.random.randint(0, 256, size=(24,320,320,1), dtype=np.int16).tostring())), \
            }

            writer = tf.python_io.TFRecordWriter(filename)
            example = tf.train.Example(features=tf.train.Features(feature=feature_list))
            writer.write(example.SerializeToString())
            writer.close()
            print("Wrote " + filename)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generates dummy data for testing')
    parser.add_argument('path', type=str, help='dest path for dummy data')
    parser.add_argument('nperclass', type=int, help='# samples per class')
    parser.add_argument('--classes', type=str, help='specific classes to generate', default=None)

    args = parser.parse_args()
    if args.classes is not None:
        classes = [k.strip() for k in args.classes.split(',')]

    generate_dummy_data(args.path, args.nperclass, classes=classes)
