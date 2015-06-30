#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 32

0~1: 课程材料首次发布、最近发布距今几天

2~5: 用户初次、上次操作此课程据今几天，持续几天，初次访问课程材料距离开课时间几天

6~13: 课程的所有用户操作课程持续时间的：平均值、标准差、最大值、最小值，以及与课程持续时间的比例

14~15: month (1-12) of the first, last event in the enrollment

16~31: 用户对课程材料的首次操作时间与课程材料发布时间的日期差的：平均值、标准差、最大值、最小值，
enrollment最后一周、倒数第二周、第一周、总体
"""


import logging
import sys
import multiprocessing as par

import numpy as np
import pandas as pd

import IO


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(__file__)


def extract(enrollment, base_date):
    logger.debug('prepare datasets ...')

    enroll_all = IO.load_enrollments()
    log_all = IO.load_logs()
    obj_all = IO.load_object()

    log_all = log_all[log_all['time'] <= base_date]
    obj_all = obj_all[obj_all['start'] <= base_date]

    logger.debug('datasets prepared')

    course_time = obj_all.groupby('course_id').agg({'start': [np.min, np.max]})
    course_time.columns = ['st_c', 'et_c']
    course_time.reset_index(inplace=True)
    # first and last event time of course
    course_t = pd.merge(log_all, enroll_all, how='left', on='enrollment_id').groupby('course_id').agg({'time': [np.min, np.max]})
    course_t.columns = ['st', 'et']
    course_t.reset_index(inplace=True)
    CT = pd.merge(enroll_t, course_time, how='left', on='course_id')
    CT.ix[(~np.isnan(CT['st_c'])) & (CT['st_c'] < CT['st']), 'st'] = CT.ix[(~np.isnan(CT['st_c'])) & (CT['st_c'] < CT['st']), 'st_c']
    CT.ix[(~np.isnan(CT['et_c'])) & (CT['et_c'] > CT['et']), 'et'] = CT.ix[(~np.isnan(CT['et_c'])) & (CT['et_c'] > CT['et']), 'et_c']
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
    ET['first_op'] = (ET['st_e'] - ET['st']).dt.days
    ET['first_month'] = ET['st_e'].dt.month
    ET['last_month'] = ET['et_e'].dt.month
    del ET['username']
    del ET['course_id']
    del ET['st']
    del ET['et']

    # 2~5: 用户初次、上次操作此课程据今几天，持续几天，初次访问课程材料距离开课时间几天
    # 14~15: month (1-12) of the first, last event in the enrollment
    XE = ET.copy()
    XE['st_e'] = (base_date - XE['st_e']).dt.days
    XE['et_e'] = (base_date - XE['et_e']).dt.days

    logger.debug('2~5, 14~15')

    return None
