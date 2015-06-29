#!/usr/local/bin/python3
# -*- coding:utf-8 -*-


import os
import gzip
import pickle as pkl

import pandas as pd

import Path


def cache(obj, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)


def fetch_cache(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    return data


def __cache__(func):
    def cached_func(file_path):
        pkl_path = file_path + '.pkl'
        data = fetch_cache(pkl_path)

        if data is None:
            data = func(file_path)

            cache(data, pkl_path)

        return data
    return cached_func


def cache_to(file_path):
    def __cache__(func):
        def cached_func(*args, **kwargs):
            pkl_path = Path.of_cache(file_path + '.pkl')
            data = fetch_cache(pkl_path)

            if data is None:
                data = func(*args, **kwargs)

                cache(data, pkl_path)

            return data
        return cached_func
    return __cache__


@__cache__
def __load_log__(file_path):
    log_set = pd.read_csv(file_path, parse_dates=['time'])
    log_set['event'] = log_set['event'].replace('nagivate', 'navigate')
    return log_set


def load_log_train():
    return __load_log__(Path.TRAIN_LOG)


def load_log_test():
    return __load_log__(Path.TEST_LOG)


@cache_to('log_all')
def load_logs():
    log_all = load_log_train().append(load_log_test(), ignore_index=True)
    log_all.sort('enrollment_id', inplace=True)
    log_all.reset_index(drop=True, inplace=True)

    return log_all


@__cache__
def __load_enrollment__(file_path):
    return pd.read_csv(file_path)


def load_enrollment_train():
    return __load_enrollment__(Path.TRAIN_ENROLL)


def load_enrollment_test():
    return __load_enrollment__(Path.TEST_ENROLL)


@cache_to('enroll_all')
def load_enrollments():
    enroll_all = load_enrollment_train().append(load_enrollment_test(),
                                                ignore_index=True)
    enroll_all.sort('enrollment_id', inplace=True)
    enroll_all.reset_index(drop=True, inplace=True)

    return enroll_all


def load_object():
    return pd.read_csv(Path.OBJECT, parse_dates=['start'], na_values=['null'])


def load_train_y():
    return pd.read_csv(Path.TRAIN_Y, header=None, names=['enrollment_id', 'y'])


@cache_to('full_dataset')
def load_full_dataset():
    return {
        'enroll': load_enrollments(),
        'log': load_logs(),
        'obj': load_object()
    }
