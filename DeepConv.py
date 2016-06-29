#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'


# Built-in Module
import os
import time
import logging
import warnings
import cPickle as pickle
from datetime import datetime
# Standard Modules
import numpy
import sklearn
import theano
import theano.tensor as T
# Custom Modules
import Scripts
import Layers

logger = logging.getLogger('main')
warnings.simplefilter("ignore", DeprecationWarning)


class DeepConv(object):
    def __init__(self, debug=False, load=False, save=False):
        self.args_debug = debug
        self.args_load = load
        self.args_save = save

        if self.args_debug:
            theano.exception_verbosity = 'high'
        if self.args_load:
            self.load()
        else:
            self.layers = None
        self.test_model = None
        self.validate_model = None
        self.train_model = None
        self.pred_model = None
        self.index, self.x, self.y = T.lscalar(), T.matrix('x'), T.ivector('y')

    def fit(self, data, labels, test_data, test_labels, learning_rate=0.1, n_epochs=250, nkerns=[20, 50], batch_size=500):
        logger.info('Initialising the classifier')
        rng = numpy.random.RandomState()
        data, labels = Scripts.shared_dataset(data_x=data, data_y=labels)
        test_data, test_labels = Scripts.shared_dataset(data_x=test_data, data_y=test_labels)
        if batch_size < 1:
            batch_size = data.get_value(borrow=True).shape[0]
        n_train_batches = data.get_value(borrow=True).shape[0]/batch_size
        n_test_batches = test_data.get_value(borrow=True).shape[0]/batch_size
        logger.info('Constructing the classifier')
        self.layers = []
        self.layers.append(Layers.PoolingLayer(
            rng,
            input=self.x.reshape((batch_size, 1, 28, 28)),
            image_shape=(batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        ))
        self.layers.append(Layers.PoolingLayer(
            rng,
            input=self.layers[-1].output,
            image_shape=(batch_size, nkerns[0], 12, 12),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        ))
        self.layers.append(Layers.HiddenLayer(
            rng,
            input=self.layers[-1].output.flatten(2),
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activation=T.tanh
        ))
        self.layers.append(Layers.LogisticRegression(
            input=self.layers[-1].output,
            n_in=500,
            n_out=10
        ))
        test_givens = {self.x: test_data[self.index * batch_size: (self.index + 1) * batch_size], self.y: test_labels[self.index * batch_size: (self.index + 1) * batch_size]}
        self.test_model = theano.function([self.index], self.layers[-1].errors(self.y), givens=test_givens)
        params = self.layers[0].params + self.layers[1].params + self.layers[2].params + self.layers[3].params
        cost = self.layers[-1].negative_log_likelihood(self.y)
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
        train_givens = {self.x: data[self.index * batch_size: (self.index + 1) * batch_size], self.y: labels[self.index * batch_size: (self.index + 1) * batch_size]}
        self.train_model = theano.function([self.index], cost, updates=updates, givens=train_givens)
        patience, patience_increase = 10000, 2
        validation_frequency = min(n_train_batches, patience / 2)
        epoch, count = 0, 0
        start_time = time.time()
        n_iters = n_epochs*n_train_batches
        logger.info("Fitting Classifier")
        logger.debug("{0} epochs, {1} batches, {2} iterations".format(n_epochs, n_train_batches, n_iters))
        while epoch < n_epochs and patience > count:
            epoch += 1
            for minibatch_index in xrange(n_train_batches):
                count = (epoch - 1) * n_train_batches + minibatch_index
                if count % 50 == 0:
                    percentage = round(100.0*count/n_iters, 2)
                    if percentage == 0:
                        time_stamp = "Null"
                    else:
                        time_stamp = datetime.utcfromtimestamp((time.time()-start_time)*(100.0/percentage)+start_time)
                    logger.info("training is {0}% complete (Completion at {1})".format(round(percentage, 2), time_stamp))
                train_cost = self.train_model(minibatch_index)
                if (count + 1) % validation_frequency == 0:
                    testlosses = [self.test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(testlosses)
                    logger.info('Test error of {0}% achieved on Epoch {1} Iteration {2}'.format(test_score*100.0, epoch, count+1))
                logger.debug("Iteration number {0}".format(count))
        logger.debug('Optimization complete.')
        logger.debug('Conducting final model testing')
        testlosses = [self.test_model(i) for i in xrange(n_test_batches)]
        test_score = numpy.mean(testlosses)
        t_taken = int((time.time()-start_time)/60.0)
        logger.info('Training Complete')
        logger.info('Test score of {0}%, training time {1}m'.format(test_score*100.0, t_taken))
        if self.args_save:
            self.save()

    def predict(self, x_data, batch_size=500):
        assert isinstance(x_data, numpy.ndarray), "input features must be a numpy array"
        assert len(x_data.shape) == 2, "it must be an array of feature vectors"
        logger.info('classifier prediction called')
        logger.debug('x_data shape: {0}'.format(x_data.shape))
        logger.debug('forming prediction function')
        x_data = Scripts.shared_dataset(data_x=x_data)
        givens = {self.x: x_data[self.index * batch_size: (self.index + 1) * batch_size]}
        pred_model = theano.function(inputs=[self.index], outputs=self.layers[-1].y_pred, givens=givens, on_unused_input='warn', allow_input_downcast=True)
        logger.debug('input shape: {0}'.format(x_data.get_value(borrow=True).shape))
        logger.info('beginning prediction on x_data')
        n_batches = x_data.get_value(borrow=True).shape[0]/batch_size
        result = []
        for batch_index in range(n_batches):
            logger.debug('processing batch {0}'.format(batch_index))
            batch_result = pred_model(batch_index)
            logger.debug('result generated')
            result = numpy.hstack((result, batch_result))
        logger.debug('output shape: {0}'.format(len(result)))
        # batch size, rows, columns, channels.
        return result

    def score(self, test_data, test_labels, batch_size=500):
        logger.info('Generating Classification Score')
        logger.debug('creating shared datasets')
        test_data, test_labels = Scripts.shared_dataset(data_x=test_data, data_y=test_labels)
        logger.debug('producing batch information')
        n_test_batches = test_data.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size
        logger.debug('generating theano functions')
        test_givens = {self.x: test_data[self.index * batch_size: (self.index + 1) * batch_size], self.y: test_labels[self.index * batch_size: (self.index + 1) * batch_size]}
        test_model = theano.function(inputs=[self.index], outputs=self.layers[-1].errors(self.y), givens=test_givens, on_unused_input='warn')
        logger.debug('producing test results')
        losses = [test_model(i) for i in range(n_test_batches)]
        return 1.0-numpy.mean(losses)

    def score_report(self, y_test, y_pred):
        scores = sklearn.metrics.classification_report(y_test, y_pred)
        logger.info("\n"+scores)

    def save(self, path="DeepConvolution.pkl"):
        path = os.path.join(os.path.split(__file__)[0], path)
        logger.info("Saving layers to {0}".format(path))
        with open(path, 'wb') as output:
            pickle.dump(self.layers, output, pickle.HIGHEST_PROTOCOL)
        logger.debug("Successfully saved")

    def load(self, path="DeepConvolution.pkl"):
        path = os.path.join(os.path.split(__file__)[0], path)
        logger.info("Loading layers from {0}".format(path))
        assert os.path.exists(path), "Specified Path is not valid"
        with open(path, "rb") as input_file:
            self.layers = pickle.load(input_file)
        logger.debug("Successfully loaded")
