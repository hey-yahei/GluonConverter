# Gluon Converter
Convert MXNet-Gluon model to Caffe.

## Supported Layers
* `Convolution` -> `Convolution`
* `BatchNorm` -> `BatchNorm` & `Scale`
* `Activation` (relu only) -> `ReLU`
* `Pooling` -> `Pooling` (MAX/AVG)
* `elemwise_add` -> `Eltwise` (ADD)
* `FullyConnected` -> `InnerProduct`
* `Flatten` -> `Flatten`
* `Concat` -> `Concat`
* `Dropout` -> `Dropout`

