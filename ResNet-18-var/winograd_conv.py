#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: winograd_conv.py
# Author: Xingyu Liu <liuxy610042@gmail.com>

import os
import tensorflow as tf
from .common import layer_register
from ..utils.argtools import shape2d, shape4d
from tensorflow.python.framework import ops
 

__all__ = ['WinogradConv']

# package_path = os.path.dirname(os.path.realpath(__file__))
# print('package_path: ',package_path)
package_path = '/home/swastik/code/fast-cnn/winograd2x2_cublas/winograd2x2_conv'
winograd2x2_conv_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_conv_op.so'))
winograd2x2_conv_grad_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_conv_grad_op.so'))

@layer_register()
def WinogradConv(x, in_channel, out_channel, mask=None, W_init=None):
    print ("9->", x)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0 * 9.0 / 32.0)

    W = tf.get_variable('W', [16, in_channel, out_channel], initializer=W_init)
    ######
    if mask is not None:
        m = tf.constant(mask)
        W = W * m
    ######

    return winograd2x2_conv(x, W)


def winograd2x2_conv(I, W):
    print ("99->",I, W)
    return winograd2x2_conv_module.winograd2x2_conv(I, W)

def winograd2x2_conv_grad(i1, i2, grad):
    print (i1, i2, grad)
    return winograd2x2_conv_grad_module.winograd2x2_conv_grad(i1, i2, grad)

@ops.RegisterShape('Winograd2x2Conv')
def _my_winograd2x2_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(5)
    shape2 = op.inputs[1].get_shape().with_rank(3)
    B = shape1.dims[1]
    nH = shape1.dims[2]
    nW = shape1.dims[3]
    H = nH * 2
    W = nW * 2
    K = shape2.dims[2]
    return [tf.TensorShape([B, H, W, K])]

@ops.RegisterGradient('Winograd2x2Conv')
def _my_matmul_grad(op, grad_output):
    input1 = op.inputs[0]
    input2 = op.inputs[1]
    grad1, grad2 = winograd2x2_conv_grad_module.winograd2x2_conv_grad(input1, input2, grad_output)
    return [grad1, grad2]



