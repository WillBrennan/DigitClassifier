#Deep Convolutional Neural Networks on MNIST

This repository features python code for a Deep Convolutional Neural Network.
This network allows its weights and biases to be saved allowing for the network
to be cached and used without having to train on start-up. This class has methods
which are identical to the main methods found in Sklearn classifiers. This means
much of those tools can be used on this.

###Quick Start

Getting the app to run is pretty easy, just clone the repo, install requirements, and then run!

```bash
# Clone the repo
git clone https://github.com/WillBrennan/MNIST && cd MNIST
# Install requirements
sudo pip install theano numpy sklearn
# Run the bot
python main_mnist.py
```
This package uses [Theano](http://deeplearning.net/software/theano/), so it can train on a GPU if the CUDA Toolkit is installed on the system. Information on the installation process can be found on their [website](https://developer.nvidia.com/cuda-downloads).

### Usage
This class can be used in much the same way as Sklearn is used. Here's a small example below,

```python
import numpy
import DeepConv

ConvNet = DeepConv.DeepConv()
data = numpy.random.rand(100, 28)
labels = numpy.zeros((100), dtype=numpy.uint8)
ConvNet.fit(data, labels)
predict_data = numpy.random.rand(20, 28)
predict_labels = ConvNet.predict(predict_data)
```

###Performance
Algorithm achieves an precision of 99.4% in approximately 300m on a MacBook Pro (without using GPU). In addition, here's a pretty confusion matrix.

![Confusion Matrix](https://github.com/WillBrennan/DigitClassifier/raw/master/src/confusion.png "Confusion Matrix")