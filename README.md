# Gluon Converter
Convert MXNet-Gluon model to Caffe.

## Supported Layers
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
