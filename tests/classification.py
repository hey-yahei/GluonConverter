#-*- coding: utf-8 -*-

import caffe
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms as T
from gluoncv.model_zoo import resnet18_v1, resnet18_v1b, resnet18_v2, \
                               vgg11_bn, \
                               mobilenet1_0, mobilenet_v2_1_0, \
                               squeezenet1_0, squeezenet1_1

import os
import numpy as np

results = {}
model_zoo = [resnet18_v1, resnet18_v1b, resnet18_v2, mobilenet1_0, vgg11_bn, squeezenet1_0, squeezenet1_1]
# model_zoo = [mobilenet_v2_1_0]    # test


def generate_caffe_model(softmax=False):
    import sys
    sys.path.append("..")
    from convert import convert_model, save_model

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for Net in model_zoo:
        print("Generate caffe model for", Net.__name__)
        net = Net(pretrained=True)
        text_net, binary_weights = convert_model(net, softmax=softmax)
        save_model(text_net, binary_weights, f"tmp/{Net.__name__}")


def test(Net, input_shape=(1,3,224,224), softmax=False):
    # input_ = np.random.uniform(size=input_shape)
    assert input_shape == (1,3,224,224)
    transformer = T.Compose([
        T.Resize(256, keep_ratio=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_ = image.imread("images/ILSVRC2012_val_00000293.jpeg")
    input_ = transformer(input_).reshape(*input_shape).asnumpy()

    caffe_net = caffe.Net(f"tmp/{Net.__name__}.prototxt", f"tmp/{Net.__name__}.caffemodel", caffe.TEST)
    caffe_net.blobs['data'].reshape(*input_shape)
    caffe_net.blobs['data'].data[...] = input_
    caffe_out = list(caffe_net.forward().values())[0]

    gluon_net = Net(pretrained=True)
    gluon_out = gluon_net(nd.array(input_))
    if softmax:
        gluon_out = nd.softmax(gluon_out)
    gluon_out = gluon_out.asnumpy()

    caffe_top5 = np.argsort(caffe_out)[0][-5:]
    gluon_top5 = np.argsort(gluon_out)[0][-5:]
    caffe_top1 = caffe_top5[-1]
    gluon_top1 = gluon_top5[-1]
    top5 = np.intersect1d(caffe_top5, gluon_top5).size
    top1 = caffe_top1 == gluon_top1
    absolute_diff = abs(gluon_out - caffe_out)
    relative_diff = (gluon_out - caffe_out) / (gluon_out + 1e-10)
    relative_diff = abs(relative_diff)
    results[Net.__name__] = (np.max(absolute_diff), np.mean(absolute_diff),
                             np.max(relative_diff), np.mean(relative_diff),
                             top5, top1)


if __name__ == "__main__":
    softmax_ = True
    generate_caffe_model(softmax=softmax_)

    for Net in model_zoo:
        test(Net, softmax=softmax_)

    for k, v in results.items():
        print(k)
        print("Absolute Difference(abs_max):", v[0])
        print("Absolute Difference(abs_mean):", v[1])
        print("Relative Difference(abs_max):", v[2])
        print("Relative Difference(abs_mean):", v[3])
        print("Top5 match:", f"{v[4]}/5")
        print("Top1 match:", f"{int(v[5])}/1")
        print()