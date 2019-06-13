# Gluon Converter
Convert MXNet-Gluon model to Caffe.

## Usage
1. Construct your gluon model, for example     
    ```python
    from gluoncv import resnet18_v1
    net = resnet18_v1(pretrained=True)
    ```
2. Convert it to caffe NetParamert, for example    
    ```python
    from convert import convert_model
    text_net, binary_weights = convert_model(net, input_shape=(1,3,224,224), softmax=False)
    ``` 
3. Save to files, for example
    ```python
    from convert import save_model
    save_model(text_net, binary_weights, prefix="tmp/resnet18_v1")
    ``` 

### How to convert SSD-like models?
Since gluoncv-SSD implement anchors and decoders with many basic operations, it is not easy to convert them to Caffe directly.    
1. Remove/Comment the codes of anchors and decoders just as `tests/models/ssd/ssd.py` do.    
2. Create a model without anchors and decoders, and load parameters from the origin model you train.    
    Take pretrained model `ssd_512_mobilenet1_0_voc` as example --     
    ```python
    from models.ssd import ssd_512_mobilenet1_0_voc
    net = ssd_512_mobilenet1_0_voc(pretrained=False)
    net.load_parameters("~/.mxnet/models/ssd_512_mobilenet1.0_voc-37c18076.params")
    ```
3. If convert it as usual, you may get a model without anchors and decoders.
4. To add `PriorBox` layers and `DetectionOutput` layers, you could     
    * Add them to `.prototxt` file manually.    
    * Or add them with my API like `tests/ssd_512_mobilenet1_0_voc.py`. I've test the converted model with [caffe-ssd](https://github.com/weiliu89/caffe/tree/ssd), [ncnn](https://github.com/Tencent/ncnn) and [Tengine](https://github.com/OAID/Tengine). Both caffe-ssd and ncnn are compatible, but Tengine doesn't work?! 

## Support Layers
* `Convolution` -> `Convolution`
* `BatchNorm` -> `BatchNorm` & `Scale`
* `Activation` (relu only) -> `ReLU`
* `Pooling` -> `Pooling` (MAX/AVG)       
    Note that computations of Pooling layer are so diffirent between Gluon and Caffe which may cause a large error.   
    But for **Global Pooling** layer, it is consistent.    
* `elemwise_add` -> `Eltwise` (ADD)
* `FullyConnected` -> `InnerProduct`
* `Flatten` -> `Flatten`
* `Concat` -> `Concat`
* `Dropout` -> `Dropout`
* `softmax` -> `Softmax`
* `transpose` -> `Permute` (caffe-ssd)
* `Reshape` -> `Reshape` (caffe-ssd)
