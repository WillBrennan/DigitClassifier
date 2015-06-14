#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# Built-in Modules
import argparse
import logging
# Standard Modules
# Custom Modules

logger = logging.getLogger('main')


def get_logger(level=logging.INFO, quiet=False, debug=False, to_file=''):
    assert level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.CRITICAL]
    logger = logging.getLogger('main')
    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    if debug:
        level = logging.DEBUG
    logger.setLevel(level=level)
    if not quiet:
        if to_file:
            fh = logging.FileHandler(to_file)
            fh.setLevel(level=level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        else:
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
    return logger


def get_args(default=None, args_string=''):
    if not default:
        default = {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--n_epochs', dest='n_epochs', default=250, type=int, help='number of training epochs to conduct')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=500, type=int, help='batch size for training')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help='display confusion matrix')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='silence the logger')
    parser.add_argument('-e', '--debug', dest='debug', action='store_true', help='set logger to debug')
    if args_string:
        args_string = args_string.split(' ')
        args = parser.parse_args(args_string)
    else:
        args = parser.parse_args()
    return args