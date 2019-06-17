#-*- coding: utf-8 -*-

from gluoncv.model_zoo import ssd_512_mobilenet1_0_voc

import sys
sys.path.append("..")
from convert import convert_ssd_model, save_model


if __name__ == "__main__":
    net = ssd_512_mobilenet1_0_voc(pretrained=True)
    text_net, binary_weights = convert_ssd_model(net, input_shape=(1,3,512,512), to_bgr=True)
    save_model(text_net, binary_weights, prefix="tmp/mssd512_voc")
