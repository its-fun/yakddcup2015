#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 172

0~107: 用户在该课程的操作数量（按event_source序对统计），enrollment最后一周、
倒数第二周、第一周、总体，占该用户在所有课程的比例，占该课程所有用户的比例

108~111: 用户有行为的课程数量，用户行为的最后一周、倒数第二周、第一周、总体

112: 课程的选课人数

113: trending slope of the weekly number of events within the enrollment

114~125: numbers of events in the last week, the first week, and the week
before the last week of the enrollment; ratio on all courses by user; ratio
on all users by course

126~129: average, standard deviation, maximal, minimal weekly numbers of
events in the enrollment period

130~131: coefficients b and c in the polynomial model y = a + bx + cx**2,
where x is week number (all start from 0), and y is the weekly number of
events

132~138: 7 counts of events in Monday to Sunday

139~162: 24 counts of events in hour 0-23

163~169: 7 counts of event types

170~171: 2 counts of source types
"""


import logging
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

import IO


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(__file__)


SE_PAIRS = ['browser-access', 'browser-page_close', 'browser-problem',
            'browser-video', 'server-access', 'server-discussion',
            'server-navigate', 'server-problem', 'server-wiki']


def extract(enrollment, base_date):
    logger.debug('prepare datasets ...')

    enroll_all = IO.load_enrollments()
    log_all = IO.load_logs()

    log_all = log_all[log_all['time'] <= base_date]
    log_all['source_event'] = log_all['source'] + '-' + log_all['event']

    log_all = pd.merge(log_all, enroll_duration(), how='left',
                       on='enrollment_id')
    log_all['day_diff'] = (log_all['time'] - log_all['st']).dt.days
    log_all['week_diff'] = log_all['day_diff'] // 7
    log_all['count'] = 1

    logger.debug('datasets prepared')

    log_last_week = log_all[
        log_all['time'] > log_all['et'] - timedelta(days=7)].copy()
    log_2nd_last_week = log_all[
        (log_all['time'] > log_all['et'] - timedelta(days=14)) &
        (log_all['time'] <= log_all['et'] - timedelta(days=7))].copy()
    log_first_week = log_all[
        log_all['time'] < log_all['st'] + timedelta(days=7)].copy()

    # 0~107: 用户在该课程的操作数量（按event_source序对统计），enrollment最后一周、
    # 倒数第二周、第一周、总体，占该用户在所有课程的比例，占该课程所有用户的比例
    # 114~125: numbers of events in the last week, the first week, and the week
    # before the last week of the enrollment; ratio on all courses by user;
    # ratio on all users by course
    EUC_last_week = count_source_event_features(log_last_week, enroll_all)
    EUC_2nd_last_week = count_source_event_features(log_2nd_last_week,
                                                    enroll_all)
    EUC_first_week = count_source_event_features(log_first_week, enroll_all)
    EUC_all = count_source_event_features(log_all, enroll_all)

    logger.debug('0~107, 114~125')

    # 108~111: 用户有行为的课程数量，用户行为的最后一周、倒数第二周、第一周、总体
    C_last_week = count_courses_by_user(log_last_week, enroll_all)
    C_2nd_last_week = count_courses_by_user(log_2nd_last_week, enroll_all)
    C_first_week = count_courses_by_user(log_first_week, enroll_all)
    C_all = count_courses_by_user(log_all, enroll_all)

    logger.debug('108~111')

    # 112: 课程的选课人数
    U_count = pd.merge(
        log_all, enroll_all, how='left', on='enrollment_id')\
        .groupby('course_id').agg({'username': lambda us: len(np.unique(us))})\
        .reset_index().rename(columns={'username': 'user_count'})

    logger.debug('112')

    return None


def count_courses_by_user(log, enroll_all):
    return pd.merge(
        log, enroll_all, how='left', on='enrollment_id')\
        .groupby('username').agg({'course_id': lambda cs: len(np.unique(cs))})\
        .reset_index().rename(columns={'course_id': 'course_count'})


def count_source_event_features(log, enroll_all):
    E, U, C = count_source_event(log, enroll_all)

    E.rename(columns=lambda c: c + '_e' if c != 'enrollment_id' else c,
             inplace=True)
    U.rename(columns=lambda c: c + '_u' if c != 'username' else c,
             inplace=True)
    C.rename(columns=lambda c: c + '_c' if c != 'course_id' else c,
             inplace=True)

    E_with_UC = pd.merge(E, enroll_all, how='left', on='enrollment_id')
    EU = pd.merge(E_with_UC, U, how='left', on='username')
    EUC = pd.merge(EU, C, how='left', on='course_id')

    EUC['count_u'] = EUC['count_e'] / EUC['count_u']
    EUC['count_c'] = EUC['count_e'] / EUC['count_c']

    for c in SE_PAIRS:
        c_e = c + '_e'
        c_u = c + '_u'
        c_c = c + '_c'
        EUC[c_u] = EUC[c_e] / EUC[c_u]
        EUC[c_c] = EUC[c_e] / EUC[c_c]

    del EUC['username']
    del EUC['course_id']

    return EUC.fillna(0)


def count_source_event(log, enroll_all):
    """
    Returns
    -------
    E: a pandas DataFrame, whose columns are:
    enrollment_id, count, browser-access, browser-page_close, browser-problem,
    browser-video, server-access, server-discussion, server-navigate,
    server-problem, server-wiki

    enrollment_id is available enrollment id
    count is total count of events
    others are counts of respective kinds of events


    U: a pandas DataFrame, whose columns are:
    username, count, browser-access, browser-page_close, browser-problem,
    browser-video, server-access, server-discussion, server-navigate,
    server-problem, server-wiki

    username is available user id
    count is total count of events of the user
    others are counts of respective kinds of events of the user


    C: a pandas DataFrame, whose columns are:
    course_id, count, browser-access, browser-page_close, browser-problem,
    browser-video, server-access, server-discussion, server-navigate,
    server-problem, server-wiki

    course_id is available course id
    count is total count of events of the course
    others are total counts of respective kinds of events of the course
    """
    counts = log.groupby(['enrollment_id', 'source_event'])\
        .agg({'count': np.sum}).reset_index()
    for se in SE_PAIRS:
        counts[se] = 0
        counts.loc[counts['source_event'] == se, se] += \
            counts[counts['source_event'] == se]['count']
    del counts['source_event']
    E = counts.groupby(['enrollment_id']).agg(np.sum).reset_index()

    E_with_UC = pd.merge(E, enroll_all, how='left', on='enrollment_id')
    U = E_with_UC.groupby('username').agg(np.sum).reset_index()
    del U['enrollment_id']

    C = E_with_UC.groupby('course_id').agg(np.sum).reset_index()
    del C['enrollment_id']

    return E, U, C


@IO.cache_to('enroll_duration')
def enroll_duration():
    log_all = IO.load_logs()
    enroll_t = log_all.groupby('enrollment_id').agg({'time': [np.min, np.max]})
    enroll_t.columns = ['st', 'et']
    return enroll_t.reset_index()
