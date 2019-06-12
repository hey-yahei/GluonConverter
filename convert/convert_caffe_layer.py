#-*- coding: utf-8 -*-
# ============================================================================
# MIT License
#
# Copyright (c) 2019 hey-yahei
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================

from . import caffe_pb2 as pb2
import numpy as np

__all__ = ['build_converters']
__author__ = 'YaHei'


_pool_type_map = {
    "max": pb2.PoolingParameter.MAX,
    "avg": pb2.PoolingParameter.AVE
}


def _as_blob(array):
    blob = pb2.BlobProto()
    blob.shape.dim.extend(array.shape)
    blob.data.extend(array.astype(float).flat)
    return blob


def _link(layer, name, bottoms, tops):
    layer.name = name
    for b in bottoms:
        layer.bottom.append(b)
    for t in tops:
        layer.top.append(t)

    return layer


def data(in_shape):
    layer = pb2.LayerParameter()
    layer.type = 'Input'
    input_shape = pb2.BlobShape()
    input_shape.dim.extend(in_shape)
    layer.input_param.shape.extend([input_shape])

    return _link(layer, "data", [], ["data"])


def convolution(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "Convolution"

    """ Parse Parameters & Attributes """
    # parameters
    assert len(params) in (1, 2), len(params)
    weight = params[0]
    bias = params[1] if len(params) == 2 else None
    # attributes
    oc, ic, kh, kw = weight.shape
    ph, pw = eval(attrs.get("pad", "(1,1)"))
    sh, sw = eval(attrs.get("stride", "(1,1)"))
    dh, dw = eval(attrs.get("dilate", "(1,1)"))
    num_group = int(attrs.get("num_group", "1"))

    """ Set Basic Attributes """
    "type(kernel_size) => google.protobuf.pyext._message.RepeatedScalarContainer"
    "type(kernel_h) => int"
    # out_channels
    layer.convolution_param.num_output = oc
    # kernel_size
    if kh == kw:
        layer.convolution_param.kernel_size.extend([kh])
    else:
        layer.convolution_param.kernel_h = kh
        layer.convolution_param.kernel_w = kw
    # stride
    if sh == sw:
        layer.convolution_param.stride.extend([sh])
    else:
        layer.convolution_param.stride_h = sh
        layer.convolution_param.stride_w = sw
    # padding
    if ph == pw:
        layer.convolution_param.pad.extend([ph])
    else:
        layer.convolution_param.pad_h = ph
        layer.convolution_param.pad_w = pw
    # dilation
    assert dh == dw
    layer.convolution_param.dilation.extend([dh])
    # group
    layer.convolution_param.group = num_group

    """ Set Parameters """
    # set
    if bias is not None:
        layer.convolution_param.bias_term = True
        layer.blobs.extend([_as_blob(weight), _as_blob(bias)])
    else:
        layer.convolution_param.bias_term = False
        layer.blobs.extend([_as_blob(weight)])

    return _link(layer, name, bottoms, tops)


def batchnorm(name, bottoms, tops, params, attrs):
    """Parse Parameters"""
    assert len(params) == 4
    gamma, beta, mean, var = params

    """ BatchNorm Layer """
    layer_bn = pb2.LayerParameter()
    layer_bn.type = "BatchNorm"
    # use_global_stats = 1 at testing phase
    layer_bn.batch_norm_param.use_global_stats = 1
    # For symbol, esp is 0010000000474974513 by default
    layer_bn.batch_norm_param.eps = eval(attrs.get("eps", "0.0010000000474974513"))
    # mean and var
    layer_bn.blobs.extend([
        _as_blob(mean),
        _as_blob(var),
        _as_blob(np.array([1.]))     # scalef
    ])

    """ Scale Layer """
    layer_scale = pb2.LayerParameter()
    layer_scale.type = "Scale"
    # gamma and beta
    layer_scale.scale_param.bias_term = True
    layer_scale.blobs.extend([_as_blob(gamma), _as_blob(beta)])

    """ Link """
    bottom_name = bottoms[0]
    _link(layer_bn, f"{bottom_name}/bn", bottoms, tops)
    _link(layer_scale, f"{bottom_name}/scale", tops, tops)

    return [layer_bn, layer_scale]


def activation(name, bottoms, tops, params, attrs):
    act_type = attrs.get("act_type")
    if act_type == "relu":
        layer = pb2.LayerParameter()
        layer.type = "ReLU"
        layer.relu_param.negative_slope = 0.

        bottom_name = bottoms[0]
        return _link(layer, f'{bottom_name}/relu', bottoms, tops)
    else:
        raise ValueError(f"Unknown act_type {act_type}")


def pooling(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "Pooling"

    """ Parse Attributes """
    # pool type
    pool_type = _pool_type_map[attrs.get("pool_type")]
    # global pool
    if attrs.get("global_pool") == "True":
        layer.pooling_param.pool = pool_type
        layer.pooling_param.global_pooling = 1
        return _link(layer, name, bottoms, tops)
    # others
    kh, kw = eval(attrs.get("kernel"))
    ph, pw = eval(attrs.get("pad", "(1,1)"))
    sh, sw = eval(attrs.get("stride", "(1,1)"))

    """ Set Attributes """
    # kernel
    if kh == kw:
        layer.pooling_param.kernel_size = kh
    else:
        layer.pooling_param.kernel_h = kh
        layer.pooling_param.kernel_w = kw
    # stride
    if sh == sw:
        layer.pooling_param.stride = sh
    else:
        layer.pooling_param.stride_h = sh
        layer.pooling_param.stride_w = sw
    # padding(ceil_mode => False)
    assert attrs['pooling_convention'] == 'valid', "Caffe only support Pooling layer with ceil_mode=False"
    if ph == pw:
        layer.pooling_param.pad = ph - 1 if sh > 1 and ph > 0 else ph
    else:
        layer.pooling_param.pad_h = ph - 1 if sh > 1 and ph > 0 else ph
        layer.pooling_param.pad_w = pw - 1 if sw > 1 and pw > 0 else pw

    return _link(layer, name, bottoms, tops)


def elemwise_add(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "Eltwise"
    return _link(layer, name, bottoms, tops)


def fully_connected(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "InnerProduct"

    """ Parse Attributes """
    num_output = int(attrs.get("num_hidden"))
    layer.inner_product_param.num_output = num_output

    """ Parse Parameters """
    # extract
    assert len(params) in (1, 2)
    weight = params[0]
    bias = params[1] if len(params) == 2 else None
    # set
    if bias is not None:
        layer.inner_product_param.bias_term = True
        layer.blobs.extend([_as_blob(weight), _as_blob(bias)])
    else:
        layer.inner_product_param.bias_term = False
        layer.blobs.extend([_as_blob(weight)])

    return _link(layer, name, bottoms, tops)


def flatten(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "Flatten"
    return _link(layer, name, bottoms, tops)


def concat(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "Concat"

    dim = int(attrs['dim'])
    layer.concat_param.axis = dim

    return _link(layer, name, bottoms, tops)


def dropout(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = "Dropout"

    drop_prob = attrs['p']
    layer.dropout_param.dropout_ratio = float(drop_prob)

    return _link(layer, name, bottoms, tops)


def softmax(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = 'Softmax'
    return _link(layer, name, bottoms, tops)


def transpose(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = 'Permute'

    orders = eval(attrs['axes'])
    layer.permute_param.order.extend(orders)

    return _link(layer, name, bottoms, tops)


def reshape(name, bottoms, tops, params, attrs):
    layer = pb2.LayerParameter()
    layer.type = 'Reshape'

    shape = eval(attrs["shape"])
    layer.reshape_param.shape.dim.extend(shape)

    return _link(layer, name, bottoms, tops)

    print(attrs)


def build_converters():
    return {
        "data": data,
        "Convolution": convolution,
        "BatchNorm": batchnorm,
        "Activation": activation,
        "Pooling": pooling,
        "elemwise_add": elemwise_add,
        "FullyConnected": fully_connected,
        "Flatten": flatten,
        "Concat": concat,
        "Dropout": dropout,
        "softmax": softmax,
        "transpose": transpose,
        "Reshape": reshape
    }
