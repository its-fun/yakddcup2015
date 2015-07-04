#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 45

0: number of 3-hour defined sessions in the enrollment

1~4: average, standard deviation, maximal, minimal numbers of events in
3-hour defined sessions in the enrollment

5~8: statistics of 3-hour defined sessions: mean, std, max, min of duration

9: number of 1-hour defined sessions in the enrollment

10~13: average, standard deviation, maximal, minimal numbers of events in
1-hour defined sessions in the enrollment

14~17: statistics of 1-hour defined sessions: mean, std, max, min of duration

18: number of 12-hour defined sessions in the enrollment

19~22: average, standard deviation, maximal, minimal numbers of events in
12-hour defined sessions in the enrollment

23~26: statistics of 12-hour defined sessions: mean, std, max, min of duration

27: number of 1-day defined sessions in the enrollment

28~31: average, standard deviation, maximal, minimal numbers of events in
1-day defined sessions in the enrollment

32~35: statistics of 1-day defined sessions: mean, std, max, min of duration

36: number of 7-day defined sessions in the enrollment

37~40: average, standard deviation, maximal, minimal numbers of events in
7-day defined sessions in the enrollment

41~44: statistics of 7-day defined sessions: mean, std, max, min of duration
"""


import logging
import sys
import os
from datetime import timedelta
import multiprocessing as par

import numpy as np
import pandas as pd

import IO
import Path
import Util


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(os.path.basename(__file__))


def __get_features__(param):
    log_all, dt = param
    return dt, sessions_of(log_all, dt)


def extract(base_date):
    pkl_path = Path.of_cache('sessions.%s.pkl' % base_date)
    X = IO.fetch_cache(pkl_path)
    if X is not None:
        logger.debug('cache hit')
        return X

    logger.debug('cache missed')
    logger.debug('prepare datasets ...')

    enroll_all = IO.load_enrollments()
    log_all = IO.load_logs()

    log_all = log_all[log_all['time'] <= base_date]

    logger.debug('datasets prepared')

    check_dataframe = Util.dataframe_checker(logger)

    n_proc = par.cpu_count()
    params = [(log_all, dt)
              for dt in [timedelta(hours=3), timedelta(hours=1),
                         timedelta(hours=12), timedelta(days=1),
                         timedelta(days=7)]]
    pool = par.Pool(processes=min(n_proc, len(params)))
    X = enroll_all
    for dt, X_ in pool.map(__get_features__, params):
        check_dataframe(X_, str(dt))
        X = pd.merge(X, X_, how='left', on='enrollment_id')

    pool.close()
    pool.join()

    del X['username']
    del X['course_id']

    X.fillna(0, inplace=True)
    check_dataframe(X, 'X')
    IO.cache(X, pkl_path)

    return X


def sessions_of(log_all, delta_t):
    def __session__(group):
        group_t = group['time'].sort(inplace=False).reset_index(drop=True)
        dt = (group_t[1:].reset_index() - group_t[:-1].reset_index())['time']
        session_break = dt > delta_t
        breaks_indices = session_break[session_break].index.values

        sessions_indices = []
        i = 0
        for b in breaks_indices:
            if b < i:
                i += 1
            else:
                sessions_indices.append((i, b))
                i = b + 1
        if i < len(group_t):
            sessions_indices.append((i, len(group_t) - 1))

        feature = [len(sessions_indices)]
        indices = ['count']

        nums_of_events = [j - i + 1 for i, j in sessions_indices]
        feature += [f(nums_of_events)
                    for f in [np.average, np.std, np.max, np.min]]
        indices += ['ec_' + i for i in ['mean', 'std', 'max', 'min']]

        sessions = pd.DataFrame(
            [(group_t[i], group_t[j]) for i, j in sessions_indices],
            columns=['st', 'et'])
        duration_ratio = (sessions['et'] - sessions['st']) / delta_t
        feature += [f(duration_ratio) for
                    f in [np.average, np.std, np.max, np.min]]
        indices += ['dr_' + i for i in ['mean', 'std', 'max', 'min']]

        return pd.Series(feature, index=indices)

    return log_all.groupby('enrollment_id').apply(__session__).reset_index()
