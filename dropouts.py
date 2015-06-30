#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 11

0~1: 课程最后十天有操作的人数，与选课人数的比例

2~5: 用户平均每个课程Dropout的次数，在该课程中Dropout的次数，与平均每个课程Dropout次数的比例；
总Dropout持续时长与课程持续时间的比例

6~10: 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：平均值、方差、最大值、最小值；
有Dropout行为的课程占用户所选课程总数的比例
"""


import logging
import sys
import os
from datetime import timedelta

import numpy as np
import pandas as pd

import IO
import Path
import Util


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(os.path.basename(__file__))


def extract(enrollment, base_date):
    pkl_path = Path.of_cache('dropouts.%s.pkl' % base_date)
    X = IO.fetch_cache(pkl_path)
    if X is not None:
        logger.debug('cache hit')
        return X

    logger.debug('cache missed')
    logger.debug('prepare datasets ...')

    enroll_all = IO.load_enrollments()
    log_all = IO.load_logs()
    obj_all = IO.load_object()

    log_all = log_all[log_all['time'] <= base_date]
    obj_all = obj_all[obj_all['start'] <= base_date]

    logger.debug('datasets prepared')

    user_count_of_course = pd.merge(
        log_all, enroll_all, how='left', on='enrollment_id')\
        .groupby('course_id').agg({'username': lambda us: len(np.unique(us))})\
        .reset_index().rename(columns={'username': 'user_count'})

    import time_related
    log_all = pd.merge(log_all, time_related.course_duration(), how='left',
                       on='enrollment_id')
    remain_log = log_all[log_all['time'] > log_all['et'] - timedelta(days=10)]
    remain_count_of_course = pd.merge(
        remain_log, enroll_all, how='left', on='enrollment_id')\
        .groupby('course_id').agg({'username': lambda us: len(np.unique(us))})\
        .reset_index().rename(columns={'username': 'remain_count'})

    D_last = pd.merge(user_count_of_course, remain_count_of_course, how='left',
                      on='course_id')
    D_last['remain_count'] = D_last['remain_count'] / D_last['user_count']

    # 0~1: 课程最后十天有操作的人数，与选课人数的比例
    logger.debug('0~1')

    def __get_dropout_count__(group):
        last_op_t = group['time'].max()
        course_et = group['et'].max()
        course_st = group['st'].min()
        group_t = group['time'].sort(inplace=False)
        dd = np.array((group_t[1:].reset_index() - group_t[:-1].reset_index())
                      ['time'].dt.days)
        dc = np.sum(dd >= 10)
        dd_sum = np.sum(dd)
        if (course_et - last_op_t).days >= 10:
            dc += 1
            dd_sum += (course_et - last_op_t).days
        return pd.Series([dc, dd_sum, dd_sum / (course_et - course_st).days],
                         index=['dropout_count', 'dt_ratio'])
    D_count = log_all.groupby('enrollment_id').apply(__get_dropout_count__)\
        .reset_index()
    D_count = pd.merge(D_count, enroll_all, how='left', on='enrollment_id')

    avg_count = D_count.groupby('username')\
        .agg({'dropout_count': np.average})\
        .rename(columns={'dropout_count': 'avg_dc'}).reset_index()
    D_count = pd.merge(D_count, avg_count, how='left', on='username')
    D_count['dc_ratio'] = D_count['dropout_count'] / D_count['avg_dc']

    # 2~5: 用户平均每个课程Dropout的次数，在该课程中Dropout的次数，
    # 与平均每个课程Dropout次数的比例；总Dropout持续时长与课程持续时间的比例
    X1 = D_count.copy()
    del X1['username']
    del X1['course_id']

    # 6~10: 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：
    # 平均值、方差、最大值、最小值；有Dropout行为的课程占用户所选课程总数的比例
    X2 = D_count.groupby('username')\
        .agg({'dt_ratio': [np.average, np.std, np.max, np.min]}).reset_index()
    DC_count = D_count[D_count['dropout_count'] > 0].groupby('username')\
        .agg({'course_id': lambda cs: len(np.unique(cs))})\
        .rename(columns={'course_id': 'd_course'}).reset_index()
    import events
    UC_count = events.count_courses_by_user(log_all, enroll_all)
    X3 = pd.merge(DC_count, UC_count, how='left', on='username')
    X3['drop_course_ratio'] = (X3['d_course'] / X3['course_count'])
    del X3['d_course']
    del X3['course_count']

    check_dataframe = Util.dataframe_checker(logger)

    check_dataframe(D_last, 'D_last')
    X = pd.merge(enrollment, D_last, how='left', on='course_id')

    check_dataframe(X1, 'X1')
    X = pd.merge(X, X1, how='left', on='enrollment_id')

    check_dataframe(X2, 'X2')
    X = pd.merge(X, X2, how='left', on='username')

    check_dataframe(X3, 'X3')
    X = pd.merge(X, X3, how='left', on='username')

    del X['enrollment_id']
    del X['username']
    del X['course_id']

    X.fillna(0, inplace=True)

    check_dataframe(X, 'X')
    X = X.as_matrix()

    IO.cache(X, pkl_path)

    return X
