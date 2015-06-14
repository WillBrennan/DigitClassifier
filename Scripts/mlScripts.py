#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'


# Built-in Modules
import os
import logging
# Standard Modules
import numpy
import theano
import theano.tensor as T
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import matplotlib.pyplot as plt
# Custom Modules

logger = logging.getLogger('main')


def get_mnist():
    path = ''.join(map(lambda i: '/'+i, os.path.abspath(__file__).split('/')[1:-2]))
    path += '/Cache'
    logger.info('Looking for Cache in {0}'.format(path))
    if not os.path.isdir(path):
        os.makedirs(path)
    assert os.path.isdir(path), 'Cache Path must exists ({0})'.format(path)
    logger.info('Fetching MNIST Data')
    result = fetch_mldata('MNIST original', data_home=path)
    logger.info('Fetching Complete')
    return result


def normalise(data):
    logger.info('normalising data')
    assert isinstance(data.data, numpy.ndarray), 'data must be an sklearn data set'
    max_data, min_data = numpy.float(numpy.max(data.data)), numpy.float(numpy.min(data.data))
    logger.debug('data maximum: {0}'.format(max_data))
    logger.debug('data minimum: {0}'.format(min_data))
    data.data = (data.data-min_data)/(max_data-min_data)
    logger.debug('data normalised, performing assertions')
    max_data, min_data = numpy.float(numpy.max(data.data)), numpy.float(numpy.min(data.data))
    assert max_data <= 1, 'normalised data must have a maximum less than one'
    assert min_data >= 0, 'noramlised data must have a minimum greater than zero'
    logger.debug('assertions correct')
    return data


def sklearn2theano(data):
    logger.info('converting from sklearn to theano data format')
    x, y = data.data, data.target
    logger.info('Splitting data set')
    x, x_test, y, y_test = train_test_split(x, y, random_state=0)
    logger.info('Data Split - {0}'.format(x_test.shape[0]/x.shape[0]))
    logger.debug('conversion complete')
    return x, x_test, y, y_test


def shared_dataset(data_x=None, data_y=None, borrow=True):
    logger.info('creating shared dataset')
    if isinstance(data_x, numpy.ndarray):
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        if not isinstance(data_y, numpy.ndarray):
            return shared_x
    if isinstance(data_y, numpy.ndarray):
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        if not isinstance(data_x, numpy.ndarray):
            return T.cast(shared_y, 'int32')
    return shared_x, T.cast(shared_y, 'int32')


def confusion_matrix(y_test, y_pred):
    logger.info('generating confusion matrix')
    cm = sklearn_confusion_matrix(y_test, y_pred)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()