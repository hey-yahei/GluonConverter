# Gluon Converter
Convert MXNet-Gluon model to Caffe.

## Requirements      
* python >= 3.6 (**f-string** is used in codes, witch is supported after 3.6)
* mxnet
* gluoncv
* numpy
* caffe(optional)

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
    **if you want to convert a ssd-like model in [gluoncv](https://github.com/dmlc/gluon-cv), please use `convert_ssd_model` API but not `convert_model`.**
3. Save to files, for example
    ```python
    from convert import save_model
    save_model(text_net, binary_weights, prefix="tmp/resnet18_v1")
    ``` 

### How do I convert a ssd-like model?    
1. To fetch attributes needed by `PriorBox` and `DetectionOutput` layers, `convert_ssd_model` will extract them from [gluon-net](https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/ssd/ssd.py#L18) and [anchors](https://github.com/dmlc/gluon-cv/blob/276ffba742d4cfe51336a76b702647c52ebb6ee0/gluoncv/model_zoo/ssd/anchor.py#L9), [box_decoder](https://github.com/dmlc/gluon-cv/blob/276ffba742d4cfe51336a76b702647c52ebb6ee0/gluoncv/nn/coder.py#L204), [class_decoder](https://github.com/dmlc/gluon-cv/blob/276ffba742d4cfe51336a76b702647c52ebb6ee0/gluoncv/nn/coder.py#L329) in it. 
2. Before convert symbols to caffemodel, fake symbols for priorbox and detction_output are added into the origin symbols.
3. Since `step` could not be extract from anchors in gluon-net, it will be setted by default in caffe(step=img_size/layer_size).

I've tested the ssd models converted from gluoncv on [caffe-ssd](https://github.com/weiliu89/caffe/tree/ssd) and [ncnn](https://github.com/Tencent/ncnn) and they works well.

## Support Layers
* `Convolution` -> `Convolution`
* `BatchNorm` -> `BatchNorm` & `Scale`
* `Activation` (relu only) -> `ReLU`
* `Pooling` -> `Pooling` (MAX/AVG)       
    Note that computations of Pooling layer are so diffirent between Gluon and Caffe which may cause some errors.   
    But for **Global Pooling** layer, it is consistent.    
* `elemwise_add` -> `Eltwise` (ADD)
* `FullyConnected` -> `InnerProduct`
* `Flatten` -> `Flatten`
* `Concat` -> `Concat`
* `Dropout` -> `Dropout`
* `softmax` -> `Softmax`
* `transpose` -> `Permute` (caffe-ssd)
* `Reshape` -> `Reshape` (caffe-ssd)
