#Deep Convolutional Neural Networks on MNIST

This repository features python code for a Deep Convolutional Neural Network.
This network allows its weights and biases to be saved allowing for the network
to be cached and used without having to train on start-up. This class has methods
which are identical to the main methods found in Sklearn classifiers. This means
much of those tools can be used on this.

###Quick Start

Getting the app to run is pretty easy, just clone the repo, install requirements, and then run! The number of threads that Theano uses can be controlled via the `OMP_NUM_THREADS` enviromental variable, below it is set to 8 threads.

```bash
# Clone the repo
git clone https://github.com/WillBrennan/MNIST && cd MNIST
# Install requirements
sudo pip install theano numpy sklearn
# Run the script on GPU
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
# Run the script on CPU
OMP_NUM_THREADS=8 python main.py
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
Dataset followed the standard 80% train - 20% test split, when prediction was conducted on the test set it achieved an accuracy of 99.4% after approximitaly 90m training on a MacBook Pro (with CUDA). In addition, here's a pretty confusion matrix.

![Confusion Matrix](https://raw.githubusercontent.com/WillBrennan/DigitClassifier/master/confusion.png "Confusion Matrix")
