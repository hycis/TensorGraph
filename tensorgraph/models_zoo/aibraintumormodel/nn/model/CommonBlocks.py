'''
#-------------------------------------------------------------------------------
Common model building blocks - TensorGraph version
Louis Lee 10-02-2019
#-------------------------------------------------------------------------------
API details:
    Version: ML_BRAIN_TUMOR_v2.0.0b
    Internal identifier: Model5b
Compnent model details:
    Version: v1.0.0b
    Internal identifier: blocks.py
#-------------------------------------------------------------------------------
'''
# Python2 compatibility
from __future__ import print_function

###############################################################################
# Common model layers & blocks                                                #
###############################################################################
import tensorflow as tf
from .....node import StartNode, HiddenNode, EndNode
from .....layers import BaseLayer, BaseModel, Template, Concat, NoChange, \
    RELU, LeakyRELU, MaxPooling3D, Dropout, Flatten
from tensorflow.python.layers.normalization import BatchNormalization as TFBatchNorm
try:
    import horovod.tensorflow as hvd
except:
    print("WARNING: Horovod module missing. No parallel MPI runs possible")
import numpy as np
import os
import sys
from math import ceil
tf.logging.set_verbosity(tf.logging.ERROR)

###############################################################################
# Primitives                                                                  #
###############################################################################
class Conv3D(BaseLayer):
    @BaseLayer.init_name_scope
    def __init__(self, num_filters=None, kernel_size=(3,3,3), stride=(1,1,1),
                 filter=None, b=None, padding='VALID', \
                 initializer=tf.contrib.layers.xavier_initializer()):
        """
        3d convolution
        Args:
            kernel_size: [filter_depth, filter_height, filter_width]
            stride: [stride_depth, stride_height, stride_width]
            filter (tensorflow.Variable): 5D Variable of shape
                ``[filter_depth, filter_height, filter_width, input_channels, num_filters]``
            b (tensorflow.Variable): 1D Variable of shape [num_filters]
            initializer (tf.initializer): initializer function for W and b
            padding: ``SAME``:
                >>> pad_along_depth = ((out_depth - 1) * stride[0] + filter_depth - in_depth)
                >>> pad_along_height = ((out_height - 1) * stride[1] + filter_height - in_height)
                >>> pad_along_width = ((out_width - 1) * stride[2] + filter_width - in_width)
                ``VALID``: padding is always 0
        """

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.filter = filter
        self.b = b
        self.initializer = initializer

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        b,d,h,w,c = state_below.shape
        c = int(c)
        self.filter_shape = self.kernel_size + (c, self.num_filters)
        if self.filter is None:
            self.filter = tf.Variable(self.initializer(self.filter_shape), \
                name=self.__class__.__name__ + '_filter')
        if self.b is None:
            self.b = tf.Variable(self.initializer([self.num_filters]), \
                name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        '''
        Args:
            state_below: (b, d, h, w, c)
        '''
        conv_out = tf.nn.conv3d(state_below, self.filter, strides=(1,)+tuple(self.stride)+(1,),
                                padding=self.padding)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.filter, self.b]

class Conv3D_Transpose(BaseLayer):
    @BaseLayer.init_name_scope
    def __init__(self, num_filters=None, kernel_size=(3,3,3), stride=(1,1,1),
                 filter=None, b=None, padding='VALID', \
                 initializer=tf.contrib.layers.xavier_initializer()):
        '''
        Args:
            filter (tensorflow.Variable): 5D Variable of shape
                ``[kernel_size[0], kernel_size[1], kernel_size[2], input_channels, num_filters]``
            b (tensorflow.Variable): 1D Variable of shape [num_filters]
            initializer (tf.initializer): initializer function for W and b
            output_shape: 3D tuple of (d, h, w)
            padding: ``SAME``:
                >>> pad_along_height = ((out_height - 1) * stride[0] + filter_height - in_height)
                >>> pad_along_width = ((out_width - 1) * stride[1] + filter_width - in_width)
                ``VALID``: padding is always 0
        '''
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.filter = filter
        self.b = b
        self.initializer = initializer

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        b,d,h,w,c = state_below.shape
        c = int(c)
        self.filter_shape = self.kernel_size + (self.num_filters, c)
        if self.filter is None:
            self.filter = tf.Variable(self.initializer(self.filter_shape), \
                name=self.__class__.__name__ + '_filter')
        if self.b is None:
            self.b = tf.Variable(self.initializer([self.num_filters]), \
                name=self.__class__.__name__ + '_b')
        assert self.padding in ['SAME','VALID'],'padding algorithm must be SAME or VALID'
        if self.padding == 'SAME':
            output_shape_d = ceil(int(d) * float(self.stride[0]))
            output_shape_h = ceil(int(h) * float(self.stride[1]))
            output_shape_w = ceil(int(w) * float(self.stride[2]))
        else:
            output_shape_d = ceil(int(d) * float(self.stride[0])) - 1 + self.kernel_size[0]
            output_shape_h = ceil(int(h) * float(self.stride[1])) - 1 + self.kernel_size[1]
            output_shape_w = ceil(int(w) * float(self.stride[2])) - 1 + self.kernel_size[2]
        self.output_shape = (output_shape_d,output_shape_h,output_shape_w)

    def _train_fprop(self, state_below):
        '''
        Args:
            state_below: (b, d, h, w, c)
        '''
        batch_size = tf.shape(state_below)[0]
        depth, height, width = self.output_shape
        deconv_shape = tf.stack((batch_size, int(depth), int(height), int(width), self.num_filters))
        conv_out = tf.nn.conv3d_transpose(value=state_below, filter=self.filter, output_shape=deconv_shape,
                                          strides=(1,)+tuple(self.stride)+(1,), padding=self.padding)
        return tf.nn.bias_add(conv_out, self.b)

    @property
    def _variables(self):
        return [self.filter, self.b]

class Linear(BaseLayer):
    @BaseLayer.init_name_scope
    def __init__(self, this_dim=None, W=None, b=None, \
        initializer=tf.contrib.layers.xavier_initializer()):
        """
        Description:
            This is a fully connected layer
        Args:
            this_dim (int): dimension of this layer
            initializer (tf.initializer): initializer function for W and b
        """
        self.this_dim = this_dim
        self.W = W
        self.b = b
        self.initializer = initializer

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        prev_dim = int(state_below.shape[1])
        if self.W is None:
            self.W = tf.Variable(self.initializer([prev_dim, self.this_dim]), \
                name=self.__class__.__name__ + '_W')
        if self.b is None:
            initializer = tf.contrib.layers.xavier_initializer()
            self.b = tf.Variable(self.initializer([self.this_dim]), \
                name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        return tf.matmul(state_below, self.W) + self.b

    @property
    def _variables(self):
        return [self.W, self.b]

class MaskedLinear(BaseLayer):
    # Different from ....'s 'LinearMasked' as the mask operation is
    # performed on W first before adding bias i.e. W' = W*mask followed by
    # out = W'*b, in order to only suppress/modify weight contributions from
    # input nodes while preserving biases at outputs
    @BaseLayer.init_name_scope
    def __init__(self, this_dim=None, W=None, b=None, mask=None, \
        initializer=tf.contrib.layers.xavier_initializer()):
        """
        Description:
            This is a fully connected layer with an applied mask for partial connections
            Mask is applied only to W i.e. y = (W _dot_ mask) _matmul_ x + b
            Mask must of dim W i.e. [input_dim, output_dim]
        Args:
            this_dim (int): dimension of this layer
            name (string): name of the layer
            W (tensor variable): Weight of 2D tensor matrix
            b (tensor variable): bias of 2D tensor matrix
            mask (numpy.ndarray or tensorflow placeholder): mask for partial connection
            of dim(W)
            initializer (tf.initializer): initializer function for W and b
        """

        self.this_dim = this_dim
        self.mask = mask
        self.W = W
        self.b = b
        self.initializer = initializer

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        prev_dim = int(state_below.shape[1])
        if self.W is None:
            self.W = tf.Variable(self.initializer([prev_dim, self.this_dim]), \
                name=self.__class__.__name__ + '_W')
        if self.b is None:
            self.b = tf.Variable(self.initializer([self.this_dim]), \
                name=self.__class__.__name__ + '_b')

    def _train_fprop(self, state_below):
        return tf.matmul(state_below, tf.multiply(self.W, self.mask)) + self.b

    @property
    def _variables(self):
        return [self.W, self.b]

class Matmul(BaseLayer):
    # Different from ....'s 'LinearMasked' as the mask operation is
    # performed on W first before adding bias i.e. W' = W*mask followed by
    # out = W'*b, in order to only suppress/modify weight contributions from
    # input nodes while preserving biases at outputs
    @BaseLayer.init_name_scope
    def __init__(self, mat=None):
        """
        Description:
            Matmul where output = state_below*mat
        Args:
            mat (tensor): mat of size [input_dim, output_dim] to be left-multiplied by state_below
        """
        self.mat = mat

    def _train_fprop(self, state_below):
        return tf.matmul(state_below, self.mat)

    @property
    def _variables(self):
        return [self.mat]

class DenseBlock3D(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, kernel_size, stride, keep_prob, growth_rate, nlayers, \
        bn_flavor='TFBatchNorm'):
        '''
        Description:
            one dense block from the densely connected CNN (Densely Connected
            Convolutional Networks https://arxiv.org/abs/1608.06993)
            Modification - block output is the conv output of the final dense
            layer w/o concatenating with the previous dense layer outputs
            i.e. {input -> dense1 -> concat(input,dense1) -> dense2 -> output}
            and not {input -> dense1 -> concat(input,dense1) -> dense2 ->
            concat(input,dense1,dense2) -> output}
        Args:
            growth_rate (int): number of filters to grow inside one denseblock
            nlayers (int): number of layers in one block, one layer refers to
                one group of batchnorm, relu and conv3d
        '''
        def _conv_layer(in_hn, concat_output=True):
            layers = []
            layers.append(Conv3D(num_filters=growth_rate, \
                kernel_size=kernel_size, stride=stride, padding='SAME'))
            layers.append(BatchNormalization(flavor=bn_flavor))
            layers.append(LeakyRELU())
            layers.append(Dropout(dropout_below=1.0-keep_prob))
            out_hn = HiddenNode(prev=[in_hn], layers=layers)
            if concat_output:
                out_hn = HiddenNode(prev=[in_hn, out_hn], input_merge_mode=Concat(axis=-1))
            return out_hn
        self.startnode = StartNode(input_vars=[None])
        in_hn = self.startnode
        for ilayer in range(nlayers):
            in_hn = _conv_layer(in_hn, concat_output=(not (ilayer+1)==nlayers))
        self.endnode = EndNode(prev=[in_hn])

class DenseBlockDoubleConv3D(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, kernel_size, stride, keep_prob, growth_rate, nlayers):
        '''
        Description:
            one dense block from the densely connected CNN (Densely Connected
            Convolutional Networks https://arxiv.org/abs/1608.06993)
            Modification 1 - block output is the conv output of the final dense
            layer w/o concatenating with the previous dense layer outputs
            i.e. {input -> dense1 -> concat(input,dense1) -> dense2 -> output}
            and not {input -> dense1 -> concat(input,dense1) -> dense2 ->
            concat(input,dense1,dense2) -> output}
            Modification 2 - Each dense layer comprises 1 double conv 'ConvBlock'
            and has no BN
        Args:
            growth_rate (int): number of filters to grow inside one denseblock
            nlayers (int): number of layers in one block, one layer refers to
                one group of batchnorm, relu and conv3d
        '''
        def _conv_layer(in_hn, concat_output=True):
            layers = []
            layers.append(ConvBlock( \
                num_filters1=growth_rate, num_filters2=growth_rate, \
                kernel_size1=kernel_size, kernel_size2=kernel_size, \
                stride1=stride, stride2=stride, keep_prob=keep_prob))
            out_hn = HiddenNode(prev=[in_hn], layers=layers)
            if concat_output:
                out_hn = HiddenNode(prev=[in_hn, out_hn], input_merge_mode=Concat(axis=-1))
            return out_hn
        self.startnode = StartNode(input_vars=[None])
        in_hn = self.startnode
        for ilayer in range(nlayers):
            in_hn = _conv_layer(in_hn, concat_output=(not (ilayer+1)==nlayers))
        self.endnode = EndNode(prev=[in_hn])

class _HorovodBatchNorm:
    '''
    Distributed Batch Normalization using Horovod to calculate correct
    batch statistics across all ranks
    Model requirement: Each rank must possess the same batchsize as
                       hvd.allreduce() simply takes average statistics over
                       all ranks
    Input: x = activations
           momentum = 1 - moving average decay rate
           epsilon  = eta for variance
           center   = whether to center batch statistics 'gamma' (can be
                      dropped if RELU is used as bias serves the same purpose)
           scale    = whether to scale batch statistics 'beta'
           training = training mode of batch normalization
           beta_initializer  = initializer for 'beta'
           gamma_initializer = initializer for 'gamma'
           moving_mean_initializer     = initializer for moving mean
           moving_variance_initializer = initializer for moving variance
           name        = name scope of this batch normalization
           return_stats = whether to return a dictionary of batch statistics
    Output: Argument #1 = batch-normalized activations
            Argument #2 = dictionary of batch statistics (if return_ma = True)
    '''
    def __init__(self, axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=tf.zeros_initializer(),
                 gamma_initializer=tf.ones_initializer(),
                 moving_mean_initializer=tf.zeros_initializer(),
                 moving_variance_initializer=tf.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None,
                 name=None):
        if renorm:
            raise NotImplementedError
        if fused:
            raise NotImplementedError

        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.moving_mean_initializer=moving_mean_initializer
        self.moving_variance_initializer=moving_variance_initializer
        self.beta_regularizer=beta_regularizer
        self.gamma_regularizer=gamma_regularizer
        self.beta_constraint=beta_constraint
        self.gamma_constraint=gamma_constraint
        self.renorm=renorm
        self.renorm_clipping=renorm_clipping
        self.renorm_momentum=renorm_momentum
        self.fused=fused
        self.name=name

    def build(self, input_shape):
        with tf.variable_scope(self.name):
            # Get # independent batch statistics required
            bn_dim = input_shape[self.axis]
            # Get list of all axes for stats reduction
            self.reduce_axes = list(range(len(input_shape)))
            del self.reduce_axes[self.axis]
            self.reduce_axes = tuple(self.reduce_axes)
            # If input has only 1 axis pass 'None' to redunction functions
            if len(self.reduce_axes) == 0:
                self.reduce_axes = None
            # batchnorm scaling
            if self.scale:
                self.gamma = tf.get_variable( \
                    'gamma', [bn_dim], dtype=tf.float32, \
                    initializer=self.gamma_initializer, \
                    regularizer=self.gamma_regularizer, \
                    constraint=self.gamma_constraint)
            else:
                self.gamma = tf.constant(1.0, tf.float32)
            # batchnorm centering
            if self.center:
                self.beta = tf.get_variable( \
                    'beta', [bn_dim], dtype=tf.float32, \
                    initializer=self.beta_initializer, \
                    regularizer=self.beta_regularizer, \
                    constraint=self.beta_constraint)
            else:
                self.beta = tf.constant(0, tf.float32)
            # moving mean & variance (for inference)
            self.moving_mean = tf.get_variable( \
                'moving_mean', [bn_dim], dtype=tf.float32, \
                initializer=self.moving_mean_initializer, trainable=False)
            self.moving_variance = tf.get_variable( \
                'moving_variance', [bn_dim], dtype=tf.float32, \
                initializer=self.moving_variance_initializer, trainable=False)

    def apply(self, x, training):
        # train/test batch statistics
        if training:
            assert 'horovod.tensorflow' in sys.modules, \
                "ERROR: HorovodBatchNorm called in train mode w/o horovod.tensorflow"
            # batch mean & variance (for training)
            batch_mean = tf.reduce_mean(x, axis=self.reduce_axes)
            batch_mean_sqr = tf.reduce_mean(x*x, axis=self.reduce_axes)

            batch_mean = hvd.allreduce(batch_mean, average=True)
            batch_mean_sqr = hvd.allreduce(batch_mean_sqr, average=True)
            batch_variance = batch_mean_sqr - batch_mean*batch_mean

            # Update moving mean & variance
            update_moving_mean = tf.assign(self.moving_mean, \
                self.moving_mean*self.momentum + batch_mean*(1.0-self.momentum), \
                name='update_moving_mean')
            update_moving_variance = tf.assign(self.moving_variance, \
                self.moving_variance*self.momentum + batch_variance*(1.0-self.momentum), \
                name='update_moving_variance')
            # Make sure ops to update  moving stats are in UPDATE_OPS collection
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

            y = tf.nn.batch_normalization(x, batch_mean, \
                batch_variance, self.beta, self.gamma, self.epsilon)
        else:
            y = tf.nn.batch_normalization(x, self.moving_mean, \
                self.moving_variance, self.beta, self.gamma, self.epsilon)
        return y

class BatchNormalization(BaseLayer):
    _bn_flavors = {'TFBatchNorm': TFBatchNorm, 'HorovodBatchNorm': _HorovodBatchNorm}
    @BaseLayer.init_name_scope
    def __init__(self, **kwargs):
        '''
        Reference:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            http://arxiv.org/abs/1502.03167
            Modification 1 - Proper name scope (prevents BN variables from existing in highest level namescope)
            Modification 2 - Added option to use Horovod BN (proper sync for batch mean/var across processes)
        Args:
            axis: Integer, the axis that should be normalized (typically the features
                axis). For instance, after a `Conv2D` layer with
                `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
            momentum: Momentum for the moving average.
            epsilon: Small float added to variance to avoid dividing by zero.
            center: If True, add offset of `beta` to normalized tensor. If False, `beta`
                is ignored.
            scale: If True, multiply by `gamma`. If False, `gamma` is
                not used. When the next layer is linear (also e.g. `nn.relu`), this can be
                disabled since the scaling can be done by the next layer.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            moving_mean_initializer: Initializer for the moving mean.
            moving_variance_initializer: Initializer for the moving variance.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: An optional projection function to be applied to the `beta`
                weight after being updated by an `Optimizer` (e.g. used to implement
                norm constraints or value constraints for layer weights). The function
                must take as input the unprojected variable and must return the
                projected variable (which must have the same shape). Constraints are
                not safe to use when doing asynchronous distributed training.
            gamma_constraint: An optional projection function to be applied to the
                `gamma` weight after being updated by an `Optimizer`.
            renorm: Whether to use Batch Renormalization
                (https://arxiv.org/abs/1702.03275). This adds extra variables during
                training. The inference is the same for either value of this parameter.
            renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
                scalar `Tensors` used to clip the renorm correction. The correction
                `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
                `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
                dmax are set to inf, 0, inf, respectively.
            renorm_momentum: Momentum used to update the moving means and standard
                deviations with renorm. Unlike `momentum`, this affects training
                and should be neither too small (which would add noise) nor too large
                (which would give stale estimates). Note that `momentum` is still applied
                to get the means and variances for inference.
            fused: if `True`, use a faster, fused implementation if possible.
                If `None`, use the system recommended implementation.
            flavor: BN flavor as string ('TFBatchNorm' for regular TF batchnorm,
                'HorovodBatchNorm' for custom Horovod BN)
        Note:
            >>> # To use this normalization, apply update ops below to update the mean and variance
            >>> from tensorflow.python.framework import ops
            >>> optimizer = tf.train.AdamOptimizer(learning_rate)
            >>> update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
            >>> with ops.control_dependencies(update_ops):
            >>>     train_op = optimizer.minimize(train_cost_sb)
        '''
        self.kwargs = kwargs
        self.flavor = 'TFBatchNorm'
        if 'flavor' in self.kwargs.keys():
            self.flavor = self.kwargs['flavor']
            del self.kwargs['flavor']
        assert self.flavor in ['TFBatchNorm', 'HorovodBatchNorm'], \
            "ERROR: Invalid batchnorm flavor given " + self.flavor

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        self.kwargs['name'] = tf.get_default_graph().get_name_scope()
        self.bn = self._bn_flavors[self.flavor](**self.kwargs)
        input_shape = [int(dim) for dim in state_below.shape[1:]]
        self.bn.build(input_shape=[None] + list(input_shape))

    def _train_fprop(self, state_below):
        return self.bn.apply(state_below, training=True)

    def _test_fprop(self, state_below):
        return self.bn.apply(state_below, training=False)

###############################################################################
# Blocks                                                                      #
###############################################################################
class MaskReduceDim(BaseLayer):
    @BaseLayer.init_name_scope
    def __init__(self, axis=-1):
        self.axis = axis

    def _train_fprop(self, state_below):
        attn_msk = 1.0 - state_below[:,:,:,:,0:1]
        return attn_msk

class MaskSoftmaxIdentity(BaseLayer):
    @BaseLayer.init_name_scope
    def __init__(self, axis=-1):
        self.axis = axis

    def _train_fprop(self, state_below):
        output_msk = tf.identity(tf.nn.softmax(state_below, axis=self.axis))
        return output_msk

class MaskSquare(BaseLayer):
    def _train_fprop(self, state_below):
        output_msk = tf.square(state_below)
        return output_msk

class ConvBlock(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, num_filters1, num_filters2, \
        kernel_size1, kernel_size2, stride1, stride2, keep_prob):
        layers = []
        layers.append(Conv3D(num_filters=num_filters1, \
            kernel_size=kernel_size1, stride=stride1, padding='SAME'))
        layers.append(LeakyRELU())
        layers.append(Conv3D(num_filters=num_filters2, \
            kernel_size=kernel_size2, stride=stride2, padding='SAME'))
        layers.append(LeakyRELU())
        layers.append(Dropout(dropout_below=1.0-keep_prob))

        self.startnode = StartNode(input_vars=[None])
        hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[hn])

class DeconvBlock(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, stride, keep_prob):
        layers = []
        layers.append(Conv3D_Transpose(num_filters=num_filters, \
            kernel_size=kernel_size, stride=stride, padding='SAME'))
        layers.append(LeakyRELU())
        layers.append(Dropout(dropout_below=1.0-keep_prob))

        self.startnode = StartNode(input_vars=[None])
        hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[hn])

# Segmentation final conv (output) block
class SegOutput(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, out_nchn, num_filters, \
        kernel_size, kernel_stride, keep_prob):
        layers = []
        layers.append(Conv3D(num_filters=num_filters, \
            kernel_size=kernel_size, stride=kernel_stride, padding='SAME'))
        layers.append(Dropout(dropout_below=1.0-keep_prob))
        layers.append(Conv3D(num_filters=out_nchn, \
            kernel_size=(1,1,1), stride=(1,1,1), padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        out_hn =  HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class ClsOutput(BaseModel):
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, kernel_stride, \
        pool_size, pool_stride, keep_prob):
        layers = []
        layers.append(Conv3D(num_filters=num_filters, kernel_size=kernel_size, \
            stride=kernel_stride, padding='SAME'))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

# Classification blocks
class ClsDenseV1(BaseModel):
    # ClsDenseC3
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):
        # BN -> Conv filtsize -> dense filtsize -> maxpool
        layers = []
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(Conv3D(num_filters=num_filters, kernel_size=kernel_size, \
            stride=(1,1,1), padding='SAME'))
        layers.append(DenseBlockDoubleConv3D(kernel_size=kernel_size, \
            stride=(1,1,1), keep_prob=keep_prob, growth_rate=num_filters, \
            nlayers=3))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class ClsDenseV2(BaseModel):
    # ClsDenseC4
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):
        # BN -> Conv filtsize -> dense filtsize -> maxpool
        layers = []
        layers.append(Conv3D(num_filters=num_filters, kernel_size=kernel_size, \
            stride=(1,1,1), padding='SAME'))
        layers.append(DenseBlockDoubleConv3D(kernel_size=kernel_size, \
            stride=(1,1,1), keep_prob=keep_prob, growth_rate=num_filters, \
            nlayers=3))
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class ClsDenseV3(BaseModel):
    # ClsDenseC5
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):
        # BN -> Conv filtsize -> dense filtsize -> maxpool
        layers = []
        layers.append(Conv3D(num_filters=num_filters, kernel_size=kernel_size, \
            stride=(1,1,1), padding='SAME'))
        layers.append(DenseBlock3D(kernel_size=kernel_size, \
            stride=(1,1,1), keep_prob=keep_prob, growth_rate=num_filters, \
            nlayers=3))
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

# Classification blocks
class ClsDenseV4(BaseModel):
    # ClsDenseC5XS
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):
        # BN -> Conv filtsize -> dense filtsize -> maxpool
        layers = []
        layers.append(Conv3D(num_filters=num_filters, kernel_size=kernel_size, \
            stride=(1,1,1), padding='SAME'))
        layers.append(DenseBlock3D(kernel_size=kernel_size, \
            stride=(1,1,1), keep_prob=keep_prob, growth_rate=num_filters, \
            nlayers=2))
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class ConvV1(BaseModel):
    # ConvCR
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, stride, keep_prob, \
        bn_flavor='TFBatchNorm'):
        layers = []
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(Conv3D(num_filters=num_filters, \
            kernel_size=kernel_size, stride=stride, padding='SAME'))
        layers.append(Dropout(dropout_below=1.0-keep_prob))

        self.startnode = StartNode(input_vars=[None])
        hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[hn])

class ConvV2(BaseModel):
    # SegConvInputC4
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, stride, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):
        layers = []
        layers.append(Conv3D(num_filters=num_filters, \
            kernel_size=kernel_size, stride=stride, padding='SAME'))
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))
        layers.append(Dropout(dropout_below=1.0-keep_prob))

        self.startnode = StartNode(input_vars=[None])
        hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[hn])

# Segmentation conv blocks
class SegConvV1(BaseModel):
    # SegConvC3
    @BaseModel.init_name_scope
    def __init__(self, num_filters1, num_filters2, \
        kernel_size1, kernel_size2, kernel_stride1, kernel_stride2, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):

        layers = []
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(ConvBlock( \
            num_filters1=num_filters1, num_filters2=num_filters1, \
            kernel_size1=kernel_size1, kernel_size2=kernel_size2, \
            stride1=kernel_stride1, stride2=kernel_stride2, keep_prob=keep_prob))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))
        layers.append(Dropout(dropout_below=1.0-keep_prob))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class SegConvV2(BaseModel):
    # SegConvC4
    @BaseModel.init_name_scope
    def __init__(self, num_filters1, num_filters2, \
        kernel_size1, kernel_size2, kernel_stride1, kernel_stride2, \
        pool_size, pool_stride, keep_prob, bn_flavor='TFBatchNorm'):

        layers = []
        layers.append(ConvBlock( \
            num_filters1=num_filters1, num_filters2=num_filters1, \
            kernel_size1=kernel_size1, kernel_size2=kernel_size2, \
            stride1=kernel_stride1, stride2=kernel_stride2, keep_prob=keep_prob))
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(MaxPooling3D(poolsize=pool_size, stride=pool_stride, \
            padding='SAME'))
        layers.append(Dropout(dropout_below=1.0-keep_prob))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class SegDeconvV1(BaseModel):
    # SegDeconvC3
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, kernel_stride, keep_prob, \
        bn_flavor='TFBatchNorm'):
        layers = []
        layers.append(BatchNormalization(flavor=bn_flavor))
        layers.append(DeconvBlock(num_filters=num_filters, \
            kernel_size=kernel_size, stride=kernel_stride, keep_prob=keep_prob))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class SegDeconvV2(BaseModel):
    # SegDeconvC4
    @BaseModel.init_name_scope
    def __init__(self, num_filters, kernel_size, kernel_stride, keep_prob, \
        bn_flavor='TFBatchNorm'):
        layers = []
        layers.append(DeconvBlock(num_filters=num_filters, \
            kernel_size=kernel_size, stride=kernel_stride, keep_prob=keep_prob))
        layers.append(BatchNormalization(flavor=bn_flavor))

        self.startnode = StartNode(input_vars=[None])
        out_hn = HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = EndNode(prev=[out_hn])

class ClsFCNV1(BaseModel):
    # ClsFCNCR
    @BaseModel.init_name_scope
    def __init__(self, fcn1_dim, fcn1_keep_prob, fcn2_dim, fcn2_keep_prob, \
        nclass1, nclass0, nclass_mat):
        # FCN 1
        layers1 = []
        layers1.append(Flatten())
        layers1.append(Linear(this_dim=fcn1_dim))
        layers1.append(LeakyRELU())
        layers1.append(Dropout(dropout_below=1.0-fcn1_keep_prob))
        # FCN 2
        layers1.append(Linear(this_dim=fcn2_dim))
        layers1.append(LeakyRELU())
        layers1.append(Dropout(dropout_below=1.0-fcn2_keep_prob))
        # FCN 3 -> Output Cls 1
        layers1.append(Linear(this_dim=nclass1))

        # FCN 3 -> Output Cls 0
        layers0 = []
        layers0.append(Matmul(mat=nclass_mat))

        self.startnode = StartNode(input_vars=[None])
        output_cls1 = HiddenNode(prev=[self.startnode], layers=layers1)
        output_cls0 = HiddenNode(prev=[output_cls1], layers=layers0)
        self.endnode = EndNode(prev=[output_cls0, output_cls1], \
            input_merge_mode=NoChange())

class ClsFCNV2(BaseModel):
    # ClsFCN
    @BaseModel.init_name_scope
    def __init__(self, fcn1_dim, fcn1_keep_prob, nclass1, nclass0, nclass_mat):
        # FCN 1
        layers1 = []
        layers1.append(Flatten())
        layers1.append(Linear(this_dim=fcn1_dim))
        layers1.append(LeakyRELU())
        layers1.append(Dropout(dropout_below=1.0-fcn1_keep_prob))
        # FCN 2 -> Output Cls 1
        layers1.append(Linear(this_dim=nclass1))

        # FCN 3 -> Output Cls 0
        layers0 = []
        layers0.append(MaskedLinear(this_dim=nclass0, mask=nclass_mat))

        self.startnode = StartNode(input_vars=[None])
        output_cls1 = HiddenNode(prev=[self.startnode], layers=layers1)
        output_cls0 = HiddenNode(prev=[output_cls1], layers=layers0)
        self.endnode = EndNode(prev=[output_cls0, output_cls1], \
            input_merge_mode=NoChange())

###############################################################################
# Misc                                                                        #
###############################################################################

def variable_summaries(var):
  # Attach a lot of summaries to a Tensor (for TensorBoard visualization)
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
