#-*- coding: utf-8 -*-

from mxnet import image, nd

from models.ssd import SSD_SETTINGS, ssd_512_mobilenet1_0_voc

import sys
sys.path.append("..")
from convert import convert_model_to_layers, layers_to_caffenet, save_model
from convert.add_caffe_layer import priorbox, detection_out
from convert.convert_caffe_layer import concat

def generate_model():
    net = ssd_512_mobilenet1_0_voc()
    net.load_parameters("tmp/ssd_512_mobilenet1.0_voc-37c18076.params", ignore_extra=True)

    normal_layers = convert_model_to_layers(net, input_shape=(1,3,512,512), to_bgr=True)

    # save_model( *layers_to_caffenet(normal_layers), prefix="tmp/mssd512_voc" )
    features_tops = (  # features_tops is extracted from the noraml_layer model above
        "mobilenet0_conv22",
        "mobilenet0_conv26",
        "expand_conv0",
        "expand_conv1",
        "expand_conv2",
        "expand_conv3"
    )
    locations = "concat1"
    confidences = "flatten12"

    caffe_net = normal_layers
    settings = SSD_SETTINGS['ssd_512_mobilenet1_0_voc']
    # Create PriorBox layers
    min_sizes = settings['sizes'][:-1]
    max_sizes = settings['sizes'][1:]
    aspect_ratios = settings['ratios']
    steps = settings['steps']
    kwargs = {
        "flip": settings['flip'],
        "clip": settings['clip'],
        "variance": settings['stds'],
        "offset": settings['offset']
    }
    for name, min_, max_, ar, st in zip(features_tops, min_sizes, max_sizes, aspect_ratios, steps):
        layer = priorbox(f"{name}_priorbox", name, f"{name}_priorbox",
                                min_size=min_, max_size=max_, aspect_ratio=ar, step=st, **kwargs)
        caffe_net.append(layer)
    caffe_net.append(
        concat("priorboxes", [f"{n}_priorbox" for n in features_tops], ["priorboxes"], [], {"dim":2})
    )
    # Create DetectionOutput layer
    kwargs = {
        "num_classes": settings['num_classes'],
        "nms_th": settings['nms_thresh'],
        "nms_topk": settings['nms_topk'],
        "topk": settings['post_nms'],
        "conf_th": settings['conf_thresh']
    }
    layer = detection_out("detection_out", [locations, confidences, "priorboxes"], "detection_out", **kwargs)
    caffe_net.append(layer)
    # Save
    text_net, binary_weights = layers_to_caffenet(caffe_net)
    save_model(text_net, binary_weights, prefix="tmp/mssd512_voc")


def test():
    import sys
    sys.path.insert(0, "/home/docker_share/caffe-ssd/python")
    import caffe   # caffe-ssd
    net = caffe.Net("tmp/mssd512_voc.prototxt", "tmp/mssd512_voc.caffemodel", caffe.TEST)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    input_ = image.imread("images/2012_000003.jpg")
    input_ = image.imresize(input_, 512, 512)
    input_ = nd.image.to_tensor(input_)
    input_ = nd.image.normalize(input_, mean=mean, std=std)

    net.blobs['data'].reshape(1,3,512,512)
    net.blobs['data'].data[...] = input_.reshape(1,3,512,512).asnumpy()
    out = net.forward()
    print(out)


if __name__ == "__main__":
    generate_model()
    # test()