'''
#------------------------------------------------------------------------------
# Cost Functions & Metrics #2
#------------------------------------------------------------------------------
# NEW version
# - Updated costAccumulator class
# - Updated weightedDiceLoss to accept flag zero_as_one that defines score=1.0
#   if both pred & act masks are zero (not tested, will raise NotImplementedError)
#------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
import sklearn.metrics

# MPI
import mpi4py.rc
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import mpi4py.MPI as MPI

###############################################################################
# Cost Functions where ypred = raw output                                     #
###############################################################################
def accuracy(ytrue, ypred, axis=-1, multihot=False, sample_weights=None, \
    normalize_sample_weights=False, scope='accuracy'):
    # axis: Specify axis containing one-hots (if multihot=False)
    # ypred: must be logits before sigmoid etc.
    with tf.name_scope(scope):
        nclass = ytrue.shape[-1].value
        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"
        assert axis==-1, \
            "ERROR: Due to change in function only axis=-1 is supported"

        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        if multihot:
            L = tf.equal(tf.cast(ytrue, tf.bool), tf.greater(ypred, 0))
            L = tf.cast(tf.reshape(L, (-1, dshape, nclass)), tf.float32)
        else:
            L = tf.equal(tf.argmax(ypred, axis=axis), \
                tf.argmax(ytrue, axis=axis))
            L = tf.cast(tf.reshape(L, (-1, dshape, 1)), tf.float32)

        if sample_weights is not None:
            tf_sweights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
            L = tf.transpose(tf.multiply(tf.cast( \
                tf.transpose(L, (2,1,0)), tf.float32), tf_sweights), (2,1,0))

        L = tf.cast(L, tf.float32)
        if normalize_sample_weights:
            sw = tf.reduce_sum(tf_sweights)
            mean_L = tf.cond(tf.greater(sw, 0), \
                lambda: tf.reduce_sum(L)/sw, lambda: 0.0)
        else:
            mean_L = tf.reduce_mean(L)

    return mean_L

def hardAccuracy(ytrue, ypred, axis=-1, threshold=0.5, sample_weights=None, \
    normalize_sample_weights=False, scope='hardAccuracy'):
    # axis: Specify axis containing one-hots
    with tf.name_scope(scope):
        nclass = ytrue.shape[-1].value
        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"
        assert axis==-1, \
            "ERROR: Due to change in function only axis=-1 is supported"

        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        ytrue = tf.cast(ytrue, tf.int32)
        ypred = tf.cast(tf.greater(tf.nn.sigmoid(ypred), threshold), tf.int32)
        L = tf.equal(ypred, ytrue)

        if sample_weights is not None:
            tf_sweights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
            L = tf.reshape(L, (-1, dshape, nclass))
            L = tf.transpose(tf.multiply(tf.cast( \
                tf.transpose(L, (2,1,0)), tf.float32), tf_sweights), (2,1,0))

        L = tf.cast(L, tf.float32)
        if normalize_sample_weights:
            sw = tf.reduce_sum(tf_sweights)
            mean_L = tf.cond(tf.greater(sw, 0), \
                lambda: tf.reduce_sum(L)/sw, lambda: 0.0)
        else:
            mean_L = tf.reduce_mean(L)

    return mean_L

def weightedRMSE(ytrue, ypred, weighting=False, weights=[1.0], \
    sample_weighting=False, sample_weights=None, calculate_softmax=True, \
    normalize_sample_weights=False, scope='weightedRMSE'):
    with tf.name_scope(scope):
        nclass = ytrue.shape[-1].value
        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"

        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        if weighting:
            weights = np.reshape(np.asarray(weights),(-1,))
            assert weights.size == nclass, "len(weights) != nclass"
            # Weight vector over channels of size (nclass)
            weights_v = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            weights_v = 1.0

        if sample_weighting:
            # Weight vector over batchsize of size (batchsize)
            weights_s = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            weights_s = 1.0

        flat_ytrue = tf.reshape(ytrue, (-1, dshape, nclass))
        flat_ypred = tf.reshape(ypred, (-1, dshape, nclass))
        if calculate_softmax:
            flat_ypred = tf.nn.softmax(flat_ypred, axis=-1)

        # Multiple each element loss by bcasting cls or sample weights
        flat_msqrd = tf.square(tf.subtract(flat_ytrue, flat_ypred))
        flat_msqrd = weights_v*flat_msqrd
        flat_msqrd = tf.transpose( \
            tf.transpose(flat_msqrd, (2,1,0))*weights_s, (2,1,0))

        if normalize_sample_weights:
            sw = tf.reduce_sum(weights_s)
            flat_msqrd = tf.cond(tf.greater(sw, 0), \
                lambda: tf.reduce_sum(flat_msqrd)/sw, lambda: 0.0)
        else:
            flat_msqrd = tf.reduce_mean(flat_msqrd)

        tf_rmse = tf.sqrt(flat_msqrd)

    return tf_rmse

def weightedFocalCrossEntropyLoss(ytrue, ypred, \
    weighting=False, weights=[1.0], sample_weighting=False, \
    sample_weights=None, focal=False, gamma=2, \
    normalize_sample_weights=False, scope='weightedFocalXEnt'):
    with tf.name_scope(scope):
        nclass = ytrue.shape[-1].value
        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"

        flat_ytrue = tf.reshape(ytrue, (-1, dshape, nclass))
        flat_ypred = tf.reshape(ypred, (-1, dshape, nclass))

        if weighting:
            weights = np.reshape(np.asarray(weights), (-1,))
            assert weights.size == nclass, "len(weights) != nclass"
            # Weight vector dim = (N) after x and reduce_sum with one-hot labels
            # weight array of shape (batchsize, dshape) or (1)
            weights_v = tf.convert_to_tensor(weights, dtype=tf.float32)
            tf_weights = tf.reduce_sum(weights_v*flat_ytrue, axis=-1)
        else:
            tf_weights = 1.0

        if sample_weighting:
            # batch sample weighting
            # weight array of shape (batchsize) or (1)
            tf_sweights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            tf_sweights = 1.0

        if focal:
            # 1-softmax vector
            # weight array of shape (batchsize, dshape) or (1)
            tf_csoftmax = tf.nn.softmax(flat_ypred, axis=-1)
            tf_dsoftmax = tf.reduce_sum((1.0 - tf_csoftmax)*flat_ytrue, axis=-1)
            # Focal loss vector
            tf_focal = tf.pow(tf_dsoftmax, gamma)
        else:
            tf_focal = 1.0

        # dim = (N) where N = # pixels (flattened from 2D array)
        # Need to reshape flat_ytrue and flat_ypred to (-1, nclass) to be accepted
        # into softmax function
        flat_ytrue = tf.reshape(flat_ytrue, (-1, nclass))
        flat_ypred = tf.reshape(flat_ypred, (-1, nclass))
        L = tf.nn.softmax_cross_entropy_with_logits( \
            logits=flat_ypred, labels=flat_ytrue)
        # Multiply weights by reshaping array to correct bcast shape for each of
        # by-batch and by-(batch, pixel) weight arrays
        L = tf.reshape(L, (-1, dshape))
        L = tf_sweights*tf.transpose(tf_focal*tf_weights*L, (1,0))
        if normalize_sample_weights:
            sw = tf.reduce_sum(tf_sweights)
            mean_L = tf.cond(tf.greater(sw, 0), \
                lambda: tf.reduce_sum(L)/sw, lambda: 0.0)
        else:
            mean_L = tf.reduce_mean(L)

    return mean_L

def weightedDiceLoss(ytrue, ypred, invarea=False, max_invsqr=1.0, \
    zero_as_one=False, weighting=False, weights=[1.0], \
    sample_weighting=False, sample_weights=None, hard=True, gamma=1.0, \
    normalize_sample_weights=False, batch_mean=True, scope='weightedDice'):
    # weighting - whether to use custom weights for each class
    # hard      - if false, ypred = softmax prob, else ypred = one-hot from
    #             ypred's softmax
    # Note: Loss is weighted over classes as 1/(area^2) to compensate for
    #       naturally large scores for big regions. Loss is also summed over
    #       all classes before taking quotient. Therefore, normally loss will
    #       almost always be < 1.0. To get normal dice loss for binary classes,
    #       set weighting=True, weights=[0,1.0] (assuming class #1 is of
    #       interest), hard=True (so one-hot overlap is computed)
    if zero_as_one:
        raise NotImplementedError("ERROR: zero_as_one not implemented")
    with tf.name_scope(scope):
        eta = 1e-5 # Avoids 1/0 if ytrue and ypred == 0
        nclass = ytrue.shape[-1].value
        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"

        # shape (nclass) or (1)
        if weighting:
            # Weight to compensate for class imbalances
            weights = np.reshape(np.asarray(weights),(-1,))
            assert weights.size == nclass, "len(weights) != nclass"

            tf_weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            tf_weights = tf.constant(1.0, dtype=tf.float32)

        # shape (batchsize) or (1)
        if sample_weighting:
            tf_sweights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            tf_sweights = tf.constant(1.0, dtype=tf.float32)

        # Must be binary 0/1 - reshaped to [batchsize, # pixels, # classes]
        flat_ytrue = tf.reshape(ytrue, (-1, dshape, nclass))

        # Softmax over channel class
        tf_ypred_softmax = tf.nn.softmax( \
            tf.reshape(ypred, (-1, dshape, nclass)), axis=-1)

        # If binary mask
        if hard:
            flat_index = tf.argmax(tf_ypred_softmax, axis=-1)
            flat_ypred = tf.one_hot(flat_index, depth=nclass)
        else:
            flat_ypred = tf.pow(tf_ypred_softmax, gamma)

        # output shape (batchsize, nclass)
        tf_prod = tf.reduce_sum(flat_ytrue*flat_ypred, axis=1)
        tf_sum = tf.reduce_sum(flat_ytrue + flat_ypred, axis=1)

        # Weight to compensate for large dice score for large areas
        # output shape (batchsize, nclass)
        if invarea:
            # shape (batchsize, nclass)
            tf_area = tf.reduce_sum(flat_ytrue, axis=1)
            # If area is too small, set its inverse weight to 0
            tf_mask = tf.greater_equal(tf_area, eta) # shape (batchsize, nclass)
            tf_area = tf.maximum(tf_area, eta) # shape (batchsize, nclass)
            tf_invsqr = tf.minimum(1.0/tf_area*tf_area, max_invsqr)
            tf_areaw = tf.cast(tf_mask, dtype=tf.float32)*tf_invsqr
        else:
            tf_areaw = tf.constant(1.0, dtype=tf.float32)

        if zero_as_one:
            tf_ones = tf.ones_like(tf_prod, dtype=tf.float32)
            tf_prod = tf.where(tf.equal(tf_sum, 0), tf_ones, tf_prod)

        # Combine invarea and cls weights
        tf_w = tf_areaw*tf_weights # shape (batchsize, nclass)

        # Multiclass Dice loss - sum over nclass sepearately for numer and denom
        # output shape = (batchsize)
        tf_dice = tf.reduce_sum(tf_w*tf_prod, axis=-1)/ \
            tf.maximum(tf.reduce_sum(tf_w*tf_sum, axis=-1), eta*eta)

        tf_dice = 1.0 - 2.0*tf_dice
        # Reduce batchsize dimension
        if batch_mean:
            if normalize_sample_weights:
                sw = tf.reduce_sum(tf_sweights)
                tf_dice = tf.cond(tf.greater(sw, 0), \
                    lambda: tf.reduce_sum(tf_dice*tf_sweights)/sw, lambda: 0.0)
            else:
                tf_dice = tf.reduce_mean(tf_dice*tf_sweights)
        else:
            assert not normalize_sample_weights, \
                "ERROR: normalize_sample_weights cannot be used with batch_mean=F"
            tf_dice = tf.reduce_sum(tf_dice*tf_sweights)

        return tf_dice

def weightedSigmoidCrossEntropyLoss(ytrue, ypred, \
    weighting=False, weights=[1.0], sample_weighting=False, \
    sample_weights=None, normalize_sample_weights=False, \
    scope='weightedSigmoidXEnt'):
    with tf.name_scope(scope):
        nclass = ytrue.shape[-1].value
        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"

        # shape (nclass)
        if weighting:
            weights = np.reshape(np.asarray(weights),(-1,))
            assert weights.size == nclass, "len(weights) != nclass"
            # Weight vector dim = (N) after x and reduce_sum with one-hot labels
            tf_weights = tf.convert_to_tensor(weights, dtype=tf.float32)
        else:
            tf_weights = 1.0

        if sigmoid_weighting:
            sigmoid_weights = np.reshape(np.asarray(sigmoid_weights),(-1,))
            assert sigmoid_weights.size == nclass, "len(weights) != nclass"
            tf_sigw = tf.convert_to_tensor(sigmoid_weights, dtype=tf.float32)
        else:
            tf_sigw = 1.0

        # shape (batchsize)
        if sample_weighting:
            tf_sweights = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            tf_sweights = 1.0

        flat_ytrue = tf.reshape(ytrue, (-1, dshape, nclass))
        flat_ypred = tf.reshape(ypred, (-1, dshape, nclass))

        flat_ytrue = tf.reshape(flat_ytrue, (-1, nclass))
        flat_ypred = tf.reshape(flat_ypred, (-1, nclass))
        if sigmoid_weighting:
            loss = tf.nn.weighted_cross_entropy_with_logits( \
                labels=flat_ytrue, logits=flat_ypred, pos_weight=tf_sigw)
        else:
            loss = tf.nn.sigmoid_cross_entropy_with_logits( \
                labels=flat_ytrue, logits=flat_ypred)

        # Apply weight factors across batchsize and nclass dims by array bcast
        loss = tf.reshape(loss, (-1, dshape, nclass))
        loss = tf.transpose(tf.transpose( \
            loss*tf_weights, (2,1,0))*tf_sweights, (2,1,0))

        # Reduce batchsize dimension
        if normalize_sample_weights:
            sw = tf.reduce_sum(tf_sweights)
            loss = tf.cond(tf.greater(sw, 0), \
                lambda: tf.reduce_sum(loss)/sw, lambda: 0.0)
        else:
            loss = tf.reduce_mean(loss)

    return loss

###############################################################################
# Cost Functions where ypred = output that will be directly used for loss     #
###############################################################################
def weightedL1(ytrue, ypred, weighting=False, weights=[1.0], \
    sample_weighting=False, sample_weights=None, \
    normalize_sample_weights=False, scope='weightedL1'):
    with tf.name_scope(scope):
        nclass = ytrue.shape[-1].value
        if len(ytrue.get_shape().as_list()) > 2:
            dshape = np.product(np.asarray(ytrue.shape[1:-1], dtype=np.int32))
        else:
            dshape = 1

        assert nclass == ypred.shape[-1].value, \
            "Error: ypred nclass /= ytrue nclass"

        if weighting:
            weights = np.reshape(np.asarray(weights), (-1,))
            assert weights.size == nclass, "len(weights) != nclass"
            # Weight vector dim = (N) after x and reduce_sum with one-hot labels
            weights_v = tf.convert_to_tensor(weights,dtype=tf.float32)
        else:
            weights_v = 1.0

        # shape (batchsize)
        if sample_weighting:
            weights_s = tf.convert_to_tensor(sample_weights, dtype=tf.float32)
        else:
            weights_s = 1.0

        flat_ytrue = tf.reshape(ytrue, (-1, dshape, nclass))
        flat_ypred = tf.reshape(ypred, (-1, dshape, nclass))

        tf_loss = tf.abs(tf.subtract(flat_ytrue, flat_ypred))
        tf_loss = tf.transpose(weights_v*tf_loss, (2,1,0))*weights_s

        if normalize_sample_weights:
            sw = tf.reduce_sum(weights_s)
            tf_loss = tf.cond(tf.greater(sw, 0), \
                lambda: tf.reduce_sum(tf_loss)/sw, lambda: 0.0)
        else:
            tf_loss = tf.reduce_mean(tf_loss)

    return tf_loss

###############################################################################
# Metrics tools                                                               #
###############################################################################
def addDict(dict1, dict2, datatype=None):
    assert set(dict1.keys()) == set(dict2.keys()), "ERROR: Key mismatch"
    for k in dict2.keys():
        dict1[k] += dict2[k]
    return dict1
dictSumOp = MPI.Op.Create(addDict, commute=True)

class costAccumulator:
    '''
    Object to gather costs from each node, compute sum or averages over all
    nodes & print them
    To use:
        1. Initialize by feeding in a costname (string): cost tensor
           (TF tensor) dict or OrderedDict e.g.
           OrderedDict([('cost01', tf_cost01), ('cost02', tf_cost02)])
        2. During each step, sess.run the object's costname:cost tensor dict
           e.g. cost_dict = sess.run(costAccumulatorObject.getTensorDict()).
           cost_dict will contain costname:cost VALUE pairs after sess.run
        3. Save the new step cost into the costAccumulator object e.g.
           costAccumulatorObject.setNewCost(cost_dict, type='dict'). The new
           cost values are stored in an internal __newcosts dict variable
        4. Sum costs from all nodes by calling
           costAccumulatorObject.MPIUpdateCostAndCount(sb_size) and feeding in
           the batchsize value. This will sum __newcosts over all nodes,
           scale it by sb_size, and finally add these values to an internal
           accumulative dictionary __costdict. sb_size will also be added to
           another internal variable __divisor
        5. To print or retreive summed costs at the end of an epoch/number of
           steps, call the appropriate methods with mean=T/F flag to indicate
           whether to return the summed averaged (over __divisor) __costdict
           values e.g. costAccumulatorObject.strCost(prefix='', mean=True)))
        6. For a new epoch/number of steps, call
           costAccumulatorObject.resetCost() to reset __costdict and __divisor
    '''
    def __init__(self, costname_tensor_dict):
        '''
        Constructor
        Input: costname_tensor_dict - dict or OrderedDict of costname (string):
               cost tensor (TF tensor) pairs, If OrderedDict, printout of
               averged or summed costs will follow costname order
        Costs are stored as regular dict internally while order of costnames
        are stored separately for printouts
        '''
        # Ordered costnames
        self.__costnames = costname_tensor_dict.keys()
        # Cost dictionaries
        self.__costdict = {str(k):0.0 for k in self.__costnames}
        self.__newcosts = {str(k):0.0 for k in self.__costnames}
        self.__costtensor = {}
        for costname, tensor in costname_tensor_dict.items():
            self.__costtensor[costname] = tensor
        # # items in batch that makes up the cost values. This is accumulated
        # during each self.MPIUpdateCostAndCount call
        self.__divisor = 0

    def getCost(self, costname=None, mean=False):
        '''
        Get value of current accumulated cost for each node
        Inputs: (Optional) costname - name of cost (string) to return. If none,
                the whole cost dict will be returned
                (Optional) mean - if T, mean of the costs will be returned
                after division with self.__divisor
        Output: Dictionary of all costs if costname=None, otherwise value of
                specific costname. If mean=True, the average values will be
                returned (after division by self.__divisor)
        '''
        if mean:
            out_dict = {k:1.0*v/max(1,self.__divisor) \
                for k,v in self.__costdict.items()}
        else:
            out_dict = self.__costdict

        if costname is None:
            return out_dict
        else:
            return out_dict[costname]

    def getTensorDict(self, costname=None):
        '''
        Returns the costnamne (string):cost tensor (TF tensor) dict (regular
        dict version of the original OrderedDict used to initialize the object)
        Input: (Optional) costname - if None, whole dict will be retunred, else
               individual costname:cost tensor pair as a dict
        Output: Whole costname:cost tensor dict or individual pair in a dict
        '''
        if costname is None:
            return self.__costtensor
        else:
            return self.__costtensor[costname]

    def getDivisor(self):
        '''
        Returns the current value of the accumulated divisor
        Input: None
        Output: Current value of the accumulated divisor
        '''
        return self.__divisor

    def setNewCost(self, input, feed_dict=None, type=None):
        for k in self.__newcosts.keys():
            self.__newcosts[k] = 0.0
        if type == 'dict':
            self.__newcosts.update(input)
        elif type == 'sess':
            self.__newcosts.update(input.run( \
                self.__costtensor, feed_dict=feed_dict))
        else:
            raise ValueError("ERROR: Invalid addCost type")

    def getNewCost(self, costname=None):
        if costname is None:
            return self.__newcosts
        else:
            return self.__newcosts[costname]

    def resetCost(self):
        self.__costdict = dict.fromkeys(self.__costdict, 0.0)
        self.__newcosts = dict.fromkeys(self.__newcosts, 0.0)
        self.__divisor = 0

    def strCost(self, prefix='', mean=False):
        if mean:
            out_dict = {k:1.0*v/max(1,self.__divisor) \
                for k,v in self.__costdict.items()}
        else:
            out_dict = self.__costdict
        printstr = ""
        for k in self.__costnames:
            printstr += (" %s%s: %7.3f" % (prefix, k, out_dict[k]))
        printstr = printstr.strip()

        return printstr

    def strNewCost(self, prefix=''):
        printstr = ""
        for k in self.__costnames:
            printstr += (" %s%s: %7.3f" % (prefix, k, out_dict[k]))
        printstr = printstr.strip()

        return printstr

    def MPIUpdateCostAndCount(self, count):
        # Scale all costs by local node's batchsize
        for k in self.__newcosts.keys():
            self.__newcosts[k] *= count
        # Allreduce count and costs
        count_all = MPI.COMM_WORLD.allreduce(count, op=MPI.SUM)
        costs_all = MPI.COMM_WORLD.allreduce(self.__newcosts, op=dictSumOp)
        # Add to accumulated costs
        self.__costdict = addDict(self.__costdict, costs_all)
        self.__divisor += count_all

def confusionMatrix(y_true, y_pred, nclass):
    return sklearn.metrics.confusion_matrix( \
        np.ravel(np.argmax(y_true, axis=-1)),
        np.ravel(np.argmax(y_pred, axis=-1)), \
        labels=range(nclass))

def makeOneHot(y, axis=-1):
    '''
    Transforms input array into one-hots along a specified axis
    Input: y - Input array e.g. softmax output
           axis - Axis from which to take one-hot
    Output: Array with the same shape as y with one-hot taken along the
            specified axis
    '''
    # Find shape of array & swap one-hot axis with the last dimension
    y_tp = list(range(len(y.shape)))
    y_tp[axis], y_tp[-1] = (y_tp[-1], y_tp[axis])
    y = np.transpose(y, y_tp)

    # Flatten array & take one-hot over the last axis
    y_flat = np.reshape(y, (-1, y.shape[-1]))
    y_argmax = np.argmax(y_flat, axis=-1)

    y_flat_oh = np.zeros(y_flat.shape, dtype=int)
    y_flat_oh[range(y_flat.shape[0]), y_argmax] = 1

    # Reshape array back to original dimension & swap back one-hot axis to
    # its original dimension
    y_flat_oh = np.reshape(y_flat_oh, y.shape)
    y_oh = np.transpose(y_flat_oh, y_tp)

    return y_oh
