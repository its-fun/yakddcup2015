#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 33

0~1: 课程材料首次发布、最近发布距今几天

2~6: 用户初次、上次操作此课程据今几天，持续几天，与课程持续时间的比例，初次访问课程材料距离开课时间几天

7~14: 课程的所有用户操作课程持续时间的：平均值、标准差、最大值、最小值，以及与课程持续时间的比例

15~16: month (1-12) of the first, last event in the enrollment

17~32: 用户对课程材料的首次操作时间与课程材料发布时间的日期差的：平均值、标准差、最大值、最小值，
enrollment最后一周、倒数第二周、第一周、总体
"""


import logging
import sys
import os
from datetime import timedelta

import numpy as np
import pandas as pd

import IO
import Util
import Path


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(os.path.basename(__file__))


def extract(enrollment, base_date):
    pkl_path = Path.of_cache('time_related.%s.pkl' % base_date)
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

    course_time = obj_all[~pd.isnull(obj_all['start'])].groupby('course_id')\
        .agg({'start': [np.min, np.max]})
    course_time.columns = ['st_c', 'et_c']
    course_time.reset_index(inplace=True)
    # first and last event time of course
    course_t = pd.merge(log_all, enroll_all, how='left', on='enrollment_id')\
        .groupby('course_id').agg({'time': [np.min, np.max]})
    course_t.columns = ['st', 'et']
    course_t.reset_index(inplace=True)
    CT = pd.merge(course_t, course_time, how='left', on='course_id')
    CT.ix[CT['st_c'] < CT['st'], 'st'] = CT.ix[CT['st_c'] < CT['st'], 'st_c']
    CT.ix[CT['et_c'] > CT['et'], 'et'] = CT.ix[CT['et_c'] > CT['et'], 'et_c']
    del CT['st_c']
    del CT['et_c']

    # 0~1: 课程材料首次发布、最近发布距今几天
    XC = CT.copy()
    XC['st'] = (base_date - XC['st']).dt.days
    XC['et'] = (base_date - XC['et']).dt.days

    logger.debug('0~1')

    ET = log_all.groupby('enrollment_id').agg({'time': [np.min, np.max]})
    ET.columns = ['st_e', 'et_e']
    ET.reset_index(inplace=True)
    ET['duration'] = (ET['et_e'] - ET['st_e']).dt.days
    ET = pd.merge(ET, enroll_all, how='left', on='enrollment_id')
    ET = pd.merge(ET, CT, how='left', on='course_id')
    ET['duration_ratio'] = (ET['et'] - ET['st']).dt.days
    ET['duration_ratio'] = ET['duration'] / ET['duration_ratio']
    ET['first_op'] = (ET['st_e'] - ET['st']).dt.days
    ET['first_month'] = ET['st_e'].dt.month
    ET['last_month'] = ET['et_e'].dt.month
    UT = ET.copy()
    del ET['st']
    del ET['et']
    del ET['username']
    del ET['course_id']

    # 2~6: 用户初次、上次操作此课程据今几天，持续几天，与课程持续时间的比例，
    # 初次访问课程材料距离开课时间几天
    # 15~16: month (1-12) of the first, last event in the enrollment
    XE = ET.copy()
    XE['st_e'] = (base_date - XE['st_e']).dt.days
    XE['et_e'] = (base_date - XE['et_e']).dt.days

    logger.debug('2~6, 15~16')

    # 7~14: 课程的所有用户操作课程持续时间的：平均值、标准差、最大值、最小值，
    # 以及与课程持续时间的比例
    XU = UT.groupby('course_id').agg({
            'duration': [np.average, np.std, np.max, np.min],
            'duration_ratio': [np.average, np.std, np.max, np.min]
            }).reset_index()

    logger.debug('7~14')

    op_time = log_all.groupby(['enrollment_id', 'object'])\
        .agg({'time': np.min}).reset_index()
    op_time = pd.merge(op_time, enroll_all, how='left', on='enrollment_id')
    op_time = pd.merge(op_time,
                       obj_all.rename(columns={'module_id': 'object'}),
                       how='left', on=['course_id', 'object'])
    op_time['delay'] = (op_time['time'] - op_time['start']).dt.days
    import events
    op_time = pd.merge(op_time, events.enroll_duration(), how='left',
                       on='enrollment_id')

    op_last_week = op_time[
        op_time['time'] > op_time['et'] - timedelta(days=7)].copy()
    op_2nd_last_week = op_time[
        (op_time['time'] > op_time['et'] - timedelta(days=14)) &
        (op_time['time'] <= op_time['et'] - timedelta(days=7))].copy()
    op_first_week = op_time[
        op_time['time'] < op_time['st'] + timedelta(days=7)].copy()

    # 17~32: 用户对课程材料的首次操作时间与课程材料发布时间的日期差的：
    # 平均值、标准差、最大值、最小值，enrollment最后一周、倒数第二周、第一周、总体
    XO_last_week = op_last_week.groupby('enrollment_id')\
        .agg({'delay': [np.nanmean, np.nanstd, np.nanmax, np.nanmin]})\
        .reset_index()
    XO_2nd_last_week = op_2nd_last_week.groupby('enrollment_id')\
        .agg({'delay': [np.nanmean, np.nanstd, np.nanmax, np.nanmin]})\
        .reset_index()
    XO_first_week = op_first_week.groupby('enrollment_id')\
        .agg({'delay': [np.nanmean, np.nanstd, np.nanmax, np.nanmin]})\
        .reset_index()
    XO_all = op_time.groupby('enrollment_id')\
        .agg({'delay': [np.nanmean, np.nanstd, np.nanmax, np.nanmin]})\
        .reset_index()

    logger.debug('17~32')

    check_dataframe = Util.dataframe_checker(logger)

    check_dataframe(XC, 'XC')
    X = pd.merge(enrollment, XC, how='left', on='course_id')

    check_dataframe(XE, 'XE')
    X = pd.merge(X, XE, how='left', on='enrollment_id')

    check_dataframe(XU, 'XU')
    X = pd.merge(X, XU, how='left', on='course_id')

    check_dataframe(XO_last_week, 'XO_last_week')
    X = pd.merge(X, XO_last_week, how='left', on='enrollment_id')

    check_dataframe(XO_2nd_last_week, 'XO_2nd_last_week')
    X = pd.merge(X, XO_2nd_last_week, how='left', on='enrollment_id')

    check_dataframe(XO_first_week, 'XO_first_week')
    X = pd.merge(X, XO_first_week, how='left', on='enrollment_id')

    check_dataframe(XO_all, 'XO_all')
    X = pd.merge(X, XO_all, how='left', on='enrollment_id')

    del X['enrollment_id']
    del X['username']
    del X['course_id']

    check_dataframe(X, 'X')
    X = X.as_matrix()

    IO.cache(X, pkl_path)

    return X
