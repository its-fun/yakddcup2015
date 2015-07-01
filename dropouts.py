#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 12

0~2: 课程最后十天有操作的人数，与选课人数的比例: 课程有更新的最后十天、课程有操作的最后十天

3~6: 用户平均每个课程Dropout的次数，在该课程中Dropout的次数，与平均每个课程Dropout次数的比例；
总Dropout持续时长与课程持续时间的比例

7~11: 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：平均值、方差、最大值、最小值；
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
    course_t1 = time_related.course_duration(log_all, obj_all, enroll_all)
    course_t2 = pd.merge(log_all, enroll_all, how='left', on='enrollment_id')\
        .groupby('course_id').agg({'time': [np.min, np.max]})
    course_t2.columns = ['st2', 'et2']
    course_t2.reset_index(inplace=True)

    log_all = pd.merge(
        pd.merge(log_all, enroll_all, how='left', on='enrollment_id'),
        course_t1, how='left', on='course_id')
    log_all = pd.merge(log_all, course_t2, how='left', on='course_id')

    remain_log1 = log_all[log_all['time'] > log_all['et'] - timedelta(days=10)]
    remain_count_of_course1 = remain_log1.groupby('course_id')\
        .agg({'username': lambda us: len(np.unique(us))})\
        .reset_index().rename(columns={'username': 'remain_count1'})

    remain_log2 = log_all[
        log_all['time'] > log_all['et2'] - timedelta(days=10)]
    remain_count_of_course2 = remain_log2.groupby('course_id')\
        .agg({'username': lambda us: len(np.unique(us))})\
        .reset_index().rename(columns={'username': 'remain_count2'})

    D_last = pd.merge(user_count_of_course, remain_count_of_course1,
                      how='left', on='course_id')
    D_last = pd.merge(D_last, remain_count_of_course2,
                      how='left', on='course_id')
    D_last.fillna(0, inplace=True)
    D_last['remain_count1'] = D_last['remain_count1'] / D_last['user_count']
    D_last['remain_count2'] = D_last['remain_count2'] / D_last['user_count']

    # 0~2: 课程最后十天有操作的人数，与选课人数的比例: 课程有更新的最后十天、课程有操作的最后十天
    logger.debug('0~2')

    def __get_dropout_count__(group):
        course_et = group['et'].max()
        course_st = group['st'].min()
        group_t = group['time'].sort(inplace=False)
        dd = np.array((group_t[1:].reset_index() - group_t[:-1].reset_index())
                      ['time'].dt.days)
        dc = np.sum(dd >= 10)
        dd_sum = np.sum(dd)
        return pd.Series([dc, dd_sum / (course_et - course_st).days],
                         index=['dropout_count', 'dt_ratio'])
    D_count = log_all.groupby('enrollment_id').apply(__get_dropout_count__)\
        .reset_index()
    D_count = pd.merge(D_count, enroll_all, how='left', on='enrollment_id')

    avg_count = D_count.groupby('username')\
        .agg({'dropout_count': np.average})\
        .rename(columns={'dropout_count': 'avg_dc'}).reset_index()
    D_count = pd.merge(D_count, avg_count, how='left', on='username')
    D_count['dc_ratio'] = D_count['dropout_count'] / D_count['avg_dc']

    # 3~6: 用户平均每个课程Dropout的次数，在该课程中Dropout的次数，
    # 与平均每个课程Dropout次数的比例；总Dropout持续时长与课程持续时间的比例
    X1 = D_count.copy()
    del X1['username']
    del X1['course_id']

    logger.debug('3~6')

    # 7~11: 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：
    # 平均值、方差、最大值、最小值；有Dropout行为的课程占用户所选课程总数的比例
    X2 = D_count.groupby('username')\
        .agg({'dt_ratio': [np.average, np.std, np.max, np.min]})\
        .reset_index()
    X2.columns = [' '.join(c).strip() for c in X2.columns.values]

    DC_count = D_count[D_count['dropout_count'] > 0].groupby('username')\
        .agg({'course_id': lambda cs: len(np.unique(cs))})\
        .rename(columns={'course_id': 'd_course'}).reset_index()
    import events
    UC_count = events.count_courses_by_user(log_all, enroll_all)
    X3 = pd.merge(DC_count, UC_count, how='left', on='username')
    X3['drop_course_ratio'] = X3['d_course'] / X3['course_count']
    del X3['d_course']
    del X3['course_count']

    logger.debug('7~11')

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
