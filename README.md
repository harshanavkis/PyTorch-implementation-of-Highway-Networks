# PyTorch implementation of Highway Networks

Implementation of Fully Connected Highway Networks found in [this](https://arxiv.org/abs/1505.00387) paper. They ease the gradient based training of very deep networks.

## Dependencies

* Python3 
* numpy==1.13.1
* torch==0.2.1+a4fc05a
* torchvision==0.1.9

### Getting started

models.py has Fully connected and Highway models for Deep Nets.

```
FcNet = models.FCModel(input_size,output_size, numLayers, hiddenDimArr, activation) #hiddenDimArr denotes the hidden layers dimensions
HfcNet = models.HighwayFcModel(inDims, input_size, output_size, numLayers, activation, gate_activation, bias) #inDims is to change the input to a desired dimension
```

After initialization to use them we just call the forward method.

```
fcOut = FcNet.forward(input)
HfcOut = HfcNet.forward(input)
```

# Defaults

The deafult initializations are:

* ReLU activation function
* Sigmoid activation for the gates
* xavier initialization of weights
* biases are initialized to -1

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
