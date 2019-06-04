#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: winograd_imtrans.py
# Author: Xingyu Liu <liuxy610042@gmail.com>

import os
import tensorflow as tf
from .common import layer_register
from ..utils.argtools import shape2d, shape4d
from tensorflow.python.framework import ops


__all__ = ['WinogradImTrans']

package_path = '/home/swastik/code/fast-cnn/winograd2x2_cublas/winograd2x2_conv'
winograd2x2_imTrans_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_imTrans_op.so'))
winograd2x2_imTrans_grad_module = tf.load_op_library(os.path.join(package_path, 'winograd2x2_imTrans_grad_op.so'))

@layer_register()
def WinogradImTrans(x, nl=tf.identity):
    return nl(winograd2x2_imTrans(x))


def winograd2x2_imTrans(I):
    return winograd2x2_imTrans_module.winograd2x2_im_trans(I)

def winograd2x2_imTrans_grad(grad):
    return winograd2x2_imTrans_grad_module.winograd2x2_im_trans_grad(grad)


@ops.RegisterShape('Winograd2x2ImTrans')
def _my_winograd2x2_shape(op):
    shape = op.inputs[0].get_shape().with_rank(4)
    H = shape.dims[1]
    W = shape.dims[2]
    nH = (H+1)/2
    nW = (W+1)/2
    return [tf.TensorShape([16, shape.dims[0], nH, nW, shape.dims[3]])]

@ops.RegisterGradient('Winograd2x2ImTrans')
def _my_matmul_grad(op, grad_output):
    # input = op.inputs[0]
    grad = winograd2x2_imTrans_grad_module.winograd2x2_im_trans_grad(grad_output)
    return [grad]