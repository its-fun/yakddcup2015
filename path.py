#! /usr/local/bin/python3
# -*- utf-8 -*-

"""
Paths of datasets
"""


import os


def of(filename):
    return os.path.abspath(os.path.join('data', filename))


OBJECT = of('object.csv')

TEST_ENROLL = of('enrollment_test.csv')
TEST_LOG = of('log_test.csv')

TRAIN_ENROLL = of('enrollment_train.csv')
TRAIN_LOG = of('log_train.csv')
TRAIN_Y = of('truth_train.csv')
