#-*- coding: utf-8 -*-

import caffe
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms as T
from gluoncv.model_zoo import resnet18_v1, resnet18_v1b, resnet18_v2, mobilenet1_0

import os
import numpy as np

results = {}
model_zoo = [resnet18_v1, resnet18_v1b, resnet18_v2, mobilenet1_0]


def generate_caffe_model():
    import sys
    sys.path.append("..")
    from convert import convert_model, save_model

    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    for Net in model_zoo:
        print("Generate caffe model for", Net.__name__)
        net = Net(pretrained=True)
        text_net, binary_weights = convert_model(net)
        save_model(text_net, binary_weights, f"tmp/{Net.__name__}")


def test(Net, input_shape=(1,3,224,224)):
    # input_ = np.random.uniform(size=input_shape)
    assert input_shape == (1,3,224,224)
    transformer = T.Compose([
        T.Resize(256, keep_ratio=True),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_ = image.imread("images/test.jpeg")
    input_ = transformer(input_).reshape(*input_shape).asnumpy()

    caffe_net = caffe.Net(f"tmp/{Net.__name__}.prototxt", f"tmp/{Net.__name__}.caffemodel", caffe.TEST)
    caffe_net.blobs['data'].reshape(*input_shape)
    caffe_net.blobs['data'].data[...] = input_
    caffe_out = list(caffe_net.forward().values())[0]

    gluon_net = Net(pretrained=True)
    gluon_out = gluon_net(nd.array(input_)).asnumpy()

    caffe_top5 = np.argsort(caffe_out)[0][-5:]
    gluon_top5 = np.argsort(gluon_out)[0][-5:]
    caffe_top1 = caffe_top5[-1]
    gluon_top1 = gluon_top5[-1]
    top5 = np.intersect1d(caffe_top5, gluon_top5).size
    top1 = caffe_top1 == gluon_top1
    diff = (gluon_out - caffe_out) / (gluon_out + 1e-10)
    diff = abs(diff)
    results[Net.__name__] = (np.max(diff), np.mean(diff), top5, top1)



if __name__ == "__main__":
    generate_caffe_model()

    for Net in model_zoo:
        test(Net)

    for k, v in results.items():
        print(k)
        print("Diff abs_max:", v[0])
        print("Diff abs_mean:", v[1])
        print(f"Top5 match: {v[2]}/5")
        print(f"Top1 match: {int(v[3])}/1")
        print()