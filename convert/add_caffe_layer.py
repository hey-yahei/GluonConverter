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

from .caffe_ssd import caffe_pb2 as pb2

__all__ = ['priorbox', 'detection_out']
__author__ = 'YaHei'


def _link(layer, name, bottoms, tops):
    layer.name = name
    for b in bottoms:
        layer.bottom.append(b)
    for t in tops:
        layer.top.append(t)

    return layer


def priorbox(name, bottom, top,
             min_size=[], max_size=[], aspect_ratio=[], flip=False, clip=False,
             variance=(0.1, 0.1, 0.2, 0.2), img_size=0, step=[], offset=0.5):
    if type(min_size) in (int, float):
        min_size = (min_size, )
    if type(max_size) in (int, float):
        max_size = (max_size, )
    if type(aspect_ratio) in (int, float):
        aspect_ratio = (aspect_ratio, )
    if type(img_size) in (int, float):
        img_size = (img_size, )
    if type(step) in (int, float):
        step = (step, )

    assert len(min_size) == len(max_size)
    assert len(variance) == 4
    assert len(img_size) in (1, 2)
    assert len(step) in (1, 2)

    layer = pb2.LayerParameter()
    layer.type = "PriorBox"

    layer.prior_box_param.min_size.extend(min_size)
    layer.prior_box_param.max_size.extend(max_size)
    layer.prior_box_param.aspect_ratio.extend(aspect_ratio)
    layer.prior_box_param.variance.extend(variance)
    layer.prior_box_param.flip = flip
    layer.prior_box_param.clip = clip
    if len(img_size) == 1:
        if img_size[0] != 0:
            layer.prior_box_param.img_size = img_size[0]
    else:
        if img_size[0] != 0 and img_size[1] != 0:
            layer.prior_box_param.img_h = img_size[0]
            layer.prior_box_param.img_w = img_size[1]
    if len(step) == 1:
        layer.prior_box_param.step = step[0]
    else:
        layer.prior_box_param.step_h = step[0]
        layer.prior_box_param.step_w = step[1]
    layer.prior_box_param.offset = offset

    return _link(layer, name, [bottom, "data"], [top])


def detection_out(name, bottoms, top,
                  num_classes, nms_th, nms_topk, topk, conf_th):
    layer = pb2.LayerParameter()
    layer.type = "DetectionOutput"

    layer.detection_output_param.num_classes = num_classes
    layer.detection_output_param.share_location = True
    layer.detection_output_param.background_label_id = 0   # The first class
    layer.detection_output_param.nms_param.nms_threshold = nms_th
    layer.detection_output_param.nms_param.top_k = nms_topk
    layer.detection_output_param.code_type = 2    # CENTER_SIZE
    layer.detection_output_param.keep_top_k = topk
    layer.detection_output_param.confidence_threshold = conf_th

    return _link(layer, name, bottoms, [top])
