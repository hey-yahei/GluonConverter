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

from mxnet import symbol

import json
import os
from google.protobuf import text_format

from .convert_caffe_layer import build_converters
from . import caffe_pb2 as pb2

__all__ = ['convert_model', 'save_model']
__author__ = 'YaHei'

_converter = build_converters()


def _extract_node_ops(sym):
    """ Construct a dictionary with (node_name, op_name) key-value paris from json """
    node_ops = {}
    sym = json.loads(sym.tojson())
    for node in sym['nodes']:
        name = node['name']
        op = node['op']
        node_ops[name] = op
    return node_ops


def _clean_name(net, name):
    """ Clear prefix and some suffixes for name """
    prefix = net._prefix
    name = name.replace(prefix, "")
    if name.endswith("_fwd_output"):
        name = name[:-len("_fwd_output")]
    elif name.endswith("_fwd"):
        name = name[:-len("_fwd")]
    elif name.endswith("_output"):
        name = name[:-len("_output")]

    return name


def _in_place(caffe_net):
    """ Set some layer as in_place mode(reset top_name). """
    in_place_types = ("BatchNorm", "ReLU")
    # Get op by node_name
    def get_op(name):
        for node in caffe_net:
            if node.name == name:
                return node.type
    # Collector for renamed tops/bottoms
    renames = {}

    """ First traversal: rename tops and collect them """
    for node in caffe_net:
        if node.type in in_place_types:
            top = node.top[0]
            bottom = node.bottom[0]
            # recursion to find the top node in collection
            while bottom in renames:
                bottom = renames[bottom]
            # rename tops
            if bottom != "data" and get_op(bottom) not in ("Eltwise", "Pooling"): # so ugly !!!
                node.top[0] = renames[top] = bottom

    """ Second traversal: rename bottoms and other tops """
    for node in caffe_net:
        for i, b in enumerate(node.bottom):
            if b in renames:
                node.bottom[i] = renames[b]
        for i, t in enumerate(node.top):
            if t in renames:
                node.top[i] = renames[t]


def convert_model(net, input_shape=(1,3,224,224), softmax=False):
    """
    Convert Gluon model to Caffe.
    :param net: mxnet.gluon.nn.HybridBlock
        Gluon net to convert.
    :param input_shape: tuple
        Shape of inputs.
    :param softmax: bool
        Add softmax for model.
    :return: (text_net, binary_weights)
        text_net: caffe_pb2.NetParameter
            Structure of net.
        binary_weights: caffe_pb2.NetParameter
            Weights of net.
    """
    # A list to collect layers
    caffe_net = []
    # Parameters from gluon model
    gluon_params = net.collect_params()

    """ Generate symbol model """
    input_ = symbol.Variable("data", shape=input_shape)
    sym = net(input_)
    if softmax:
        sym = symbol.SoftmaxOutput(sym)

    """ Convert data layer """
    convert_fn = _converter.get("data")
    layer = convert_fn(input_shape)
    caffe_net.append(layer)

    """ Convert other layers """
    node_ops = _extract_node_ops(sym)
    for node in sym.get_internals():
        # Basic attributes: name & op
        name = node.name
        op = node_ops[name]
        # Collect all children: inputs and parameters
        in_sym = node.get_children()
        if in_sym is None:  # data layer
            continue
        # Collectors for bottoms and parameters
        bottoms = []
        params = []
        for s in in_sym:
            s_name = s.name
            if s_name != 'data' and node_ops[s_name] == 'null':     # Parameters
                params.append(gluon_params[s_name].data().asnumpy())
            else:   # Inputs
                bottoms.append(_clean_name(net, s_name))
        # Collector for tops
        tops = [_clean_name(net, out_name) for out_name in node.list_outputs()]
        # Get convert function
        convert_fn = _converter.get(op, None)
        assert convert_fn is not None, f"unknwon op: {op}"
        # Convert gluon layer to caffe and add to collector `caffe_net`
        attrs = node.list_attr()
        layer = convert_fn(_clean_name(net, name), bottoms, tops, params, attrs)
        if op == "BatchNorm":       # BatchNorm is converted into BatchNorm & Scale
            caffe_net.extend(layer)
        else:   # Other layers
            caffe_net.append(layer)

    """ Set ReLU & BatchNorm inplace """
    _in_place(caffe_net)

    """ Caffe input """
    text_net = pb2.NetParameter()
    if os.environ.get("T2C_DEBUG"):
        text_net.debug_info = True

    """ Caffe layer parameters """
    binary_weights = pb2.NetParameter()
    binary_weights.CopyFrom(text_net)
    for layer in caffe_net:
        binary_weights.layer.extend([layer])

        layer_proto = pb2.LayerParameter()
        layer_proto.CopyFrom(layer)
        del layer_proto.blobs[:]
        text_net.layer.extend([layer_proto])

    return text_net, binary_weights


def save_model(text_net, binary_weights, prefix):
    """
    Save caffe model
    :param text_net: caffe_pb2.NetParameter
        Text net generated by `convert_model` function.
    :param binary_weights: caffe_pb2.NetParameter
        Weights generated by `convert_model` function.
    :param prefix: str
        Prefix for caffe model, `text_net` will be saved in <prefix>.prototxt while `binary_weights` <prefix>.caffemodel
    :return: None
    """
    with open(f'{prefix}.prototxt', 'w') as f:
        f.write(text_format.MessageToString(text_net))
    with open(f'{prefix}.caffemodel', 'wb') as f:
        f.write(binary_weights.SerializeToString())