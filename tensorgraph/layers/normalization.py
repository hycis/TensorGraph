import tensorflow as tf
from tensorflow.python.layers.normalization import BatchNormalization as TFBatchNorm
from .template import BaseLayer
from tensorflow.python.ops import init_ops

class L2_Normalize(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, dim):
        '''dim (int or list of ints): dimension to normalize'''
        self.dim = dim

    def _train_fprop(self, state_below):
        return tf.nn.l2_normalize(state_below, dim=self.dim)


class BatchNormalization(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer=init_ops.zeros_initializer(),
                 gamma_initializer=init_ops.ones_initializer(),
                 moving_mean_initializer=init_ops.zeros_initializer(),
                 moving_variance_initializer=init_ops.ones_initializer(),
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None):
        '''
        Reference:
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            http://arxiv.org/abs/1502.03167

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

        Note:
            >>> # To use this normalization, apply update ops below to update the mean and variance
            >>> from tensorflow.python.framework import ops
            >>> optimizer = tf.train.AdamOptimizer(learning_rate)
            >>> update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
            >>> with ops.control_dependencies(update_ops):
            >>>     train_op = optimizer.minimize(train_cost_sb)

        '''
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

    @BaseLayer.init_name_scope
    def __init_var__(self, state_below):
        scope_ = tf.get_default_graph().get_name_scope()
        self.bn = TFBatchNorm(axis=self.axis,
                              momentum=self.momentum,
                              epsilon=self.epsilon,
                              center=self.center,
                              scale=self.scale,
                              beta_initializer=self.beta_initializer,
                              gamma_initializer=self.gamma_initializer,
                              moving_mean_initializer=self.moving_mean_initializer,
                              moving_variance_initializer=self.moving_variance_initializer,
                              beta_regularizer=self.beta_regularizer,
                              gamma_regularizer=self.gamma_regularizer,
                              beta_constraint=self.beta_constraint,
                              gamma_constraint=self.gamma_constraint,
                              renorm=self.renorm,
                              renorm_clipping=self.renorm_clipping,
                              renorm_momentum=self.renorm_momentum,
                              fused=self.fused,
                              name=str(scope_))
        input_shape = [int(dim) for dim in state_below.shape[1:]]
        self.bn.build(input_shape=[None] + list(input_shape))

    def _train_fprop(self, state_below):
        return self.bn.apply(state_below, training=True)

    def _test_fprop(self, state_below):
        return self.bn.apply(state_below, training=False)


class LRN(BaseLayer):

    @BaseLayer.init_name_scope
    def __init__(self, depth_radius=None, bias=None, alpha=None, beta=None):
        '''
        Description:
            Local Response Normalization.
            The 4-D input tensor is treated as a 3-D array of 1-D vectors
            (along the last dimension), and each vector is normalized independently.
            Within a given vector, each component is divided by the weighted,
            squared sum of inputs within depth_radius. In detail,

            >>> sqr_sum[a, b, c, d] =   sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
            >>> output = input / (bias + alpha * sqr_sum) ** beta

        Args:
            depth_radius (optional int): defaults to 5. 0-D. Half-width of the 1-D normalization window.
            bias (optional float): Defaults to 1. An offset (usually positive to avoid dividing by 0).
            alpha (optional float): Defaults to 1. A scale factor, usually positive.
            beta (optional float): Defaults to 0.5. An exponent.
        '''
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta


    def _train_fprop(self, state_below):
        return tf.nn.local_response_normalization(state_below, depth_radius=self.depth_radius,
                                                  bias=self.bias, alpha=self.alpha,
                                                  beta=self.beta, name=None)
