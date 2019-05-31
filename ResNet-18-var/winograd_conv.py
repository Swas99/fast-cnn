import tensorflow as tf
from .common import layer_register
from ..utils.argtools import shape2d, shape4d
 

__all__ = ['WinogradConv']


@layer_register()
def WinogradConv(x, in_channel, out_channel, mask=None, W_init=None):

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0 * 9.0 / 32.0)

    W = tf.get_variable('W', [16, in_channel, out_channel], initializer=W_init)
    ######
    if mask is not None:
        m = tf.constant(mask)
        W = W * m
    ######

    return winograd2x2_conv(x, W)

package_path = os.path.dirname(os.path.realpath(__file__))
print('package_path: ',package_path)
winograd2x2_conv_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_conv_op.so'))
winograd2x2_conv_grad_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_conv_grad_op.so'))

def winograd2x2_conv(I, W):
	return winograd2x2_conv_module.winograd2x2_conv(I, W)

def winograd2x2_conv_grad(i1, i2, grad):
 	return winograd2x2_conv_grad_module.winograd2x2_conv_grad(i1, i2, grad)