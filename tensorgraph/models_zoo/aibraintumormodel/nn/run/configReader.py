'''
#------------------------------------------------------------------------------
# Reads INI-type config file for training
# NEW - adds adc value to series_kprob default & train_rec
#------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

try:
    import ConfigParser as configparser
except:
    import configparser

import ast
import numpy as np

class parameters:
    # Keys with default values
    __dict = { \
        'testing'      : False, \
        'model_module' : "", \
        'model_scope'  : "", \
        'use_water_mask' : True, \
        'data_module'  : "", \
        'data_dir'     : "", \
        'anatomies'    : [], \
        'biopsy_only'  : False, \
        'mask_only'    : False, \
        'water_mask'   : False, \
        'clsmatcoeff'  : (0.0, 1.0), \
        'series_kprob' : [1.0, 1.0, 1.0, 1.0], \
        'series_val'   : [1, 1, 1, 1], \
        'train_valid_seed' : 1234, \
        'radiomics'    : False, \
        'continuation' : False, \
        'restore'      : False, \
        'restore_seg'  : False, \
        'restore_cls'  : False, \
        'save'         : False, \
        'batchsize'        : 1, \
        'keep_prob'        : 1.0, \
        'keep_prob_fcn_D1' : 1.0, \
        'keep_prob_fcn_D2' : 1.0, \
        'optimizer'            : 'adam', \
        'max_epochs'           : 1, \
        'train_rec'            : False, \
        'train_loc'            : False, \
        'train_seg'            : True, \
        'water_seg'            : True, \
        'comb_seg'             : True, \
        'train_cls'            : True, \
        'big_cls'              : True, \
        'small_cls'            : True, \
        'rec_loss_coefficient' : 1.0, \
        'loc_loss_coefficient' : 1.0, \
        'seg_loss_coefficient' : 1.0, \
        'cls_loss_coefficient' : 1.0, \
        'min_learning_rate'         : 0.0001, \
        'max_learning_rate'         : 0.0002, \
        'learning_rate_decay_step'  : 10000, \
        'learning_rate_decay_rate'  : 0.999, \
        'learning_rate_epochsize'   : 2, \
        'learning_range_decay'      : False, \
        'learning_range_decay_rate' : 0.5, \
        'l2_regularizer'        : False, \
        'l2_weight_decay'       : 0.001, \
        'distributed_batchnorm' : False, \
        'batch_renorm' : False, \
        'renorm_rmax'  : [1.0, 3.0, 5000, 40000], \
        'renorm_dmax'  : [0.0, 5.0, 5000, 25000], \
        'report_every_nsteps'   : 1, \
        'save_every_nsteps'     : 10, \
        'validate_every_nepoch' : 1, \
        'save_out_every_nepoch' : 1, \
        'sel_threshold'         : [0.0,1.0], \
        'save_path'    : "", \
        'restore_path' : "", \
        'log_path'     : "", \
        'out_res_path' : "", \
        'out_res_frac' : 1.0, \
        'validation_list' : "", \
    }

    def __init__(self, config_filepath):
        try:
            parser = configparser.ConfigParser(allow_no_value=True)
            parser.read(config_filepath)
        except:
            print("ERROR: Error reading file " + config_filepath)
            raise

        assert 'train' in parser.sections(), "ERROR: Section 'train' not found"

        for key, value in self.__dict.items():
            # Parse & cast flags
            try:
                # Parse
                parser_value = parser.get('train', key)
                try:
                    # Read
                    parser_value = ast.literal_eval(parser_value)
                    # Cast
                    try:
                        value_type = type(value[0])
                    except:
                        value_type = type(value)
                    if value_type == type(""):
                        parser_value = str(parser_value).strip()
                    else:
                        parser_value = np.asarray(parser_value).astype(value_type)
                    if str(parser_value) == "":
                        parser_value = None

                except:
                    print("ERROR: Cannot convert argument for " + key)
                    raise
            except:
                if value == "":
                    parser_value = None
                else:
                    parser_value = value

            # Update dict with parsed flags
            self.__dict[key] = parser_value

        # anatomies
        try:
            if list(self.__dict['anatomies']) != []:
                self.__dict['anatomies'] = \
                    list(np.asarray(self.__dict['anatomies'], dtype=int))
        except:
            print("ERROR: Error reading anatomies")
            raise

        # sel_threshold
        try:
            self.__dict['sel_threshold'] = \
                list(np.asarray(self.__dict['sel_threshold'], dtype=float))
        except:
            print("ERROR: Error reading sel_threshold")
            raise

        # series_val (supposed to be on/off only)
        try:
            sval = np.asarray(self.__dict['series_val'], dtype=int)
            self.__dict['series_val'] = list(sval.astype(bool).astype(float))
        except:
            print("ERROR: Error reading series_val")
            raise

        # clsmatcoeff
        try:
            self.__dict['clsmatcoeff'] = \
                list(np.asarray(self.__dict['clsmatcoeff'], dtype=float))
        except:
            print("ERROR: Error reading clsmatcoeff")
            raise

        # series_kprob
        try:
            self.__dict['series_kprob'] = \
                list(np.asarray(self.__dict['series_kprob'], dtype=float))
        except:
            print("ERROR: Error reading series_kprob")
            raise

        # train_valid_seed
        self.__dict['train_valid_seed'] = int(self.__dict['train_valid_seed'])

        # renorms
        try:
            self.__dict['renorm_rmax'] = \
                list(np.asarray(self.__dict['renorm_rmax'], dtype=float))
            self.__dict['renorm_rmax'][2] = int(self.__dict['renorm_rmax'][2])
            self.__dict['renorm_rmax'][3] = int(self.__dict['renorm_rmax'][3])
        except:
            print("ERROR: Error reading renorm_rmax")
            raise
        try:
            self.__dict['renorm_dmax'] = \
                list(np.asarray(self.__dict['renorm_dmax'], dtype=float))
            self.__dict['renorm_dmax'][2] = int(self.__dict['renorm_dmax'][2])
            self.__dict['renorm_dmax'][3] = int(self.__dict['renorm_dmax'][3])
        except:
            print("ERROR: Error reading renorm_dmax")
            raise

        # Convert dict keys to actual variables
        for key, value in self.__dict.items():
            try:
                setattr(self, key, value)
            except:
                print("ERROR: Error converting flag " + key + " to variable")
                raise

    def listFlags(self):
        for key, value in sorted(self.__dict.items()):
            print(key + ": " + str(value))

    def size(self):
        return len(self.__dict)

    def dict(self):
        return self.__dict

    def setDelim(self, delim):
        self.__delim = delim

    # For MPI4PY
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, in_dict):
        for key in in_dict.keys():
            self.__dict[key] = in_dict[key]
        self.__dict__ = in_dict
