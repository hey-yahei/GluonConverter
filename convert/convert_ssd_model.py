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

from .convert_model import convert_model
from .fake_symbol import FakeSymbol

from mxnet import symbol

__all__ = ['convert_ssd_model']
__author__ = 'YaHei'


def _extract_ssd_attrs(net, anchors=None, bbox_decoder=None, cls_decoder=None):
    if anchors is None:
        anchors = net.anchor_generators
    if bbox_decoder is None:
        bbox_decoder = net.bbox_decoder
    if cls_decoder is None:
        cls_decoder = net.cls_decoder

    priorbox_attrs = []
    for an in anchors:
        attr = {
            "min_size": an._sizes[0],
            "max_size": an._sizes[1] ** 2 / an._sizes[0],   # https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/ssd/anchor.py#L38
            "aspect_ratio": an._ratios,
            "flip": False,
            "clip": an._clip,
            "variance": bbox_decoder._stds,
            # "step":
            "offset": .5
        }
        priorbox_attrs.append(attr)

    detection_out_attrs = {
        "num_classes": cls_decoder._fg_class + 1,
        "share_location": True,
        "background_label": 0,
        "nms_threshold": net.nms_thresh,
        "nms_top_k": net.nms_topk,
        "code_type": 2,    # CENTER_SIZE,
        "keep_top_k": net.post_nms,
        "confidence_threshold": cls_decoder._thresh
    }

    return priorbox_attrs, detection_out_attrs


def _find_symbol_by_bottomname(sym, bottomname):
    for s in sym.get_internals():
        children = s.get_children()
        if children is not None:
            for c in children:
                if bottomname == c.name:
                    return s


def _find_symbol_by_name(sym, name):
    for s in sym.get_internals():
        if s.name == name:
            return s


def convert_ssd_model(net, input_shape=(1,3,512,512), to_bgr=False, merge_bn=True):
    """
    Convert SSD-like model to Caffe.
    :param net: mxnet.gluon.nn.HybridBlock
        Gluon net to convert.
    :param input_shape: tuple
        Shape of inputs.
    :param to_bgr: bool
        Convert input_type from RGB to BGR.
    :param merge_bn: bool
        Merge BatchNorm and Scale layers to Convolution layers.
    :return: (text_net, binary_weights)
        text_net: caffe_pb2.NetParameter
            Structure of net.
        binary_weights: caffe_pb2.NetParameter
            Weights of net.
    """
    """ Create symbols """
    in_ = symbol.Variable("data", shape=input_shape)
    __, scores_sym, __ = net(in_)

    """ Add symbols about box_predictors and cls_predictors """
    # box_predictors
    box_pred_name = net.box_predictors[0].predictor.name
    box_transpose = _find_symbol_by_bottomname(scores_sym, f"{box_pred_name}_fwd")
    box_flatten = _find_symbol_by_bottomname(scores_sym, box_transpose.name)
    box_concat = _find_symbol_by_bottomname(scores_sym, box_flatten.name)
    # cls_prodictors
    cls_pred_name = net.class_predictors[0].predictor.name
    cls_transpose = _find_symbol_by_bottomname(scores_sym, f"{cls_pred_name}_fwd")
    cls_flatten = _find_symbol_by_bottomname(scores_sym, cls_transpose.name)
    cls_concat = _find_symbol_by_bottomname(scores_sym, cls_flatten.name)
    cls_reshape = _find_symbol_by_bottomname(scores_sym, cls_concat.name)
    cls_softmax = symbol.softmax(cls_reshape, axis=2)
    cls_flatten = symbol.flatten(cls_softmax)

    """ Collect attributes needed by Priorbox and DetectionOutput layers """
    priorbox_attrs, detection_out_attrs = _extract_ssd_attrs(net)

    """ Create fake symbol for Priorbox layers """
    priorboxes = []
    for i, box_pred in enumerate(net.box_predictors):
        pred_sym = _find_symbol_by_name(scores_sym, f"{box_pred.predictor.name}_fwd")
        # (ugly) Get Convolution symbol of predictor
        for c in pred_sym.get_children():
            if c.get_children() is not None:
                conv = c
                break
        # Create a new fake symbol for Priorbox
        priorbox = FakeSymbol(conv, name=f"{conv.name}_priorbox", _op="PriorBox", **priorbox_attrs[i])
        priorboxes.append(priorbox)
    # Concat outputs of Priorbox symbol
    pbox_concat = symbol.concat(*priorboxes, dim=2)

    """ Create fake symbol for DetectionOutput layer """
    detection_out = FakeSymbol(box_concat, cls_flatten, pbox_concat,
                               _in_num=3, name="detection_out", _op="DetectionOutput", **detection_out_attrs)

    return convert_model(net, detection_out, input_shape=input_shape, to_bgr=to_bgr, merge_bn=merge_bn)

