#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Generate datasets for training and validating, and load dataset of testing.
"""


from datetime import datetime, timedelta
import logging
import sys
import os

import numpy as np

import features
import path
import fio

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')


def load_test():
    """
    Load dataset for testing.


    Returns
    -------
    X: numpy ndarray, shape: (num_of_enrollments, num_of_features)
    Rows of features.
    """
    pkl_path = path.of_cache('test_X.pkl')
    X = fio.fetch_cache(pkl_path)
    if X is None:
        enroll_set = fio.load_enrollment_test()
        log = fio.load_logs()
        obj = fio.load_object()
        base_date = datetime(2014, 8, 1, 22, 0, 47)
        X = None
        for f in features.METHODS:
            X_ = f(enroll_set, log, obj, base_date)
            if X is None:
                X = X_
            else:
                X = np.c_[X, X_]

        fio.cache(X, pkl_path)

    return X.as_matrix()[:, 1:].astype(np.float)


def __enroll_ids_with_log__(enroll_ids, base_date):
    log = fio.load_logs()
    log_eids = set(log[log['time'] <= base_date]['enrollment_id'].unique())
    return np.array([eid for eid in enroll_ids if eid in log_eids])


def __load_dataset__(enroll_ids, base_date):
    """get all instances in this time window"""
    enroll_set = fio.load_enrollments().set_index('enrollment_id')\
        .ix[enroll_ids].reset_index()
    log = fio.load_logs()
    obj = fio.load_object()

    X = None
    for f in features.METHODS:
        X_ = f(enroll_set, log, obj, base_date)
        if X is None:
            X = X_
        else:
            X = np.c_[X, X_]

    active_eids = set(log[(log['time'] > base_date) &
                          (log['time'] <= base_date + timedelta(days=10))]
                         ['enrollment_id'])
    y = [int(eid not in active_eids) for eid in enroll_ids]

    return X.as_matrix()[:, 1:].astype(np.float), np.array(y, dtype=np.int)


def load_train(depth=0):
    """
    Load dataset for training and validating.

    *NOTE*  If you need a validating set, you SHOULD split from training set
    by yourself.


    Args
    ----
    depth: int, 0 by default
    Maximum moves of time window. 0 means no need to move time window.


    Returns
    -------
    X: numpy ndarray, shape: (num_of_enrollments, num_of_features)
    Rows of features. It is the features of all time if cache_only is True.

    y: numpy ndarray, shape: (num_of_enrollments,)
    Vector of labels. It is the labels of all time if cache_only is True.
    """
    logger = logging.getLogger('load_train')

    enroll_set = fio.load_enrollment_train()
    base_date = datetime(2014, 8, 1, 22, 0, 47)

    logger.debug('loading features before %s', base_date)

    enroll_ids = __enroll_ids_with_log__(enroll_set['enrollment_id'],
                                         base_date)

    pkl_X_path = path.of_cache('train_X.%s.pkl' % base_date)
    pkl_y_path = path.of_cache('train_y.%s.pkl' % base_date)

    X = fio.fetch_cache(pkl_X_path)
    y = fio.fetch_cache(pkl_y_path)

    if X is None or y is None:
        logger.debug('cache missed, calculating ...')

        X, _ = __load_dataset__(enroll_ids, base_date)
        y_with_id = fio.load_train_y()

        if not np.all(y_with_id[:, 0] == enroll_ids):
            logger.fatal('something wrong with enroll_ids')
            raise RuntimeError('something wrong with enroll_ids')

        y = y_with_id[:, 1]

        fio.cache(X, pkl_X_path)
        fio.cache(y, pkl_y_path)

    base_date = datetime(2014, 7, 22, 22, 0, 47)
    Dw = timedelta(days=7)

    enroll_ids = __enroll_ids_with_log__(enroll_ids, base_date)
    for _ in range(depth):
        if enroll_ids.size <= 0:
            break

        logger.debug('loading features before %s', base_date)

        # get instances and labels
        pkl_X_path = path.of_cache('train_X.%s' % base_date)
        pkl_y_path = path.of_cache('train_y.%s' % base_date)

        X_temp = fio.fetch_cache(pkl_X_path)
        y_temp = fio.fetch_cache(pkl_y_path)
        if X_temp is None or y_temp is None:
            logger.debug('cache missed, calculating ...')

            X_temp, y_temp = __load_dataset__(enroll_ids, base_date)

            fio.cache(X_temp, pkl_X_path)
            fio.cache(y_temp, pkl_y_path)

        # update instances and labels
        X = np.r_[X, X_temp]
        y = np.append(y, y_temp)

        # update base_date and enroll_ids
        base_date -= Dw
        enroll_ids = __enroll_ids_with_log__(enroll_ids, base_date)

    return X, y


if __name__ == '__main__':
    import glob
    if sys.argv[1] == 'clean':
        cached_files = glob.glob(path.of_cache('train_X*.pkl'))
        cached_files += glob.glob(path.of_cache('train_y*.pkl'))
        cached_files += glob.glob(path.of_cache('test_X.pkl'))
        for f in cached_files:
            print('Removing %s ...' % f)
            os.remove(f)

    elif sys.argv[1] == 'gen':
        X, y = load_train(depth=1)
        print('X.shape: %d x %d' % X.shape)
        print('y.shape: %d' % y.shape)
        X_test = load_test()
        print('X_test.shape: %d x %d' % X_test.shape)
