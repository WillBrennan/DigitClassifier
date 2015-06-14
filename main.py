#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'


# Built-in Modules
# Standard Modules
import theano
# Custom Modules
import Scripts
import DeepConv


if __name__ == '__main__':
    args = Scripts.get_args()
    logger = Scripts.get_logger(quiet=args.quiet, debug=args.debug)
    data = Scripts.get_mnist()
    data = Scripts.normalise(data)
    x, x_test, y, y_test = Scripts.sklearn2theano(data)
    classifier = DeepConv.DeepConv(args)
    classifier.fit(data=x, labels=y, test_data=x_test, test_labels=y_test, n_epochs=args.n_epochs, batch_size=args.batch_size)
    y_pred = classifier.predict(x_test)
    classifier.score_report(y_test=y_test, y_pred=y_pred)
    logger.info('Classifier Scoring: {0}'.format(classifier.score(x_test, y_test)))
    Scripts.confusion_matrix(y_test, y_pred)
