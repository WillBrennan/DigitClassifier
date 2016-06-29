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
This class can be used in much the same way as Sklearn is used. Here's a small example below, its easier to just call `main.py` however.

```python
import numpy
import Scripts
import DeepConv

args_save = False
args_load = False
args_debug = False
args_n_epochs = 250
args_batch_size = 500

data = Scripts.get_mnist()
data = Scripts.normalise(data)
x, x_test, y, y_test = Scripts.sklearn2theano(data)
classifier = DeepConv.DeepConv(save=args_save, load=args_load, debug=args_debug)
classifier.fit(data=x, labels=y, test_data=x_test, test_labels=y_test, n_epochs=args_n_epochs, batch_size=args_batch_size)
y_pred = classifier.predict(x_test)
classifier.score_report(y_test=y_test, y_pred=y_pred)
logger.info('Classifier Scoring: {0}'.format(classifier.score(x_test, y_test)))
Scripts.confusion_matrix(y_test, y_pred)
```

###Performance
Dataset followed the standard 80% train - 20% test split, when prediction was conducted on the test set it achieved an accuracy of 99.4% after approximitaly 90m training on a MacBook Pro (with CUDA). In addition, here's a pretty confusion matrix.

![Confusion Matrix](https://raw.githubusercontent.com/WillBrennan/DigitClassifier/master/confusion.png "Confusion Matrix")
