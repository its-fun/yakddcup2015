#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 163

0~107: 用户在该课程的操作数量（按event_source序对统计），enrollment最后一周、
倒数第二周、第一周、总体，占该用户在所有课程的比例，占该课程所有用户的比例

108~111: 用户有行为的课程数量，用户行为的最后一周、倒数第二周、第一周、总体

112: 课程的选课人数

113: trending slope of the weekly number of events within the enrollment

114~116: numbers of events in the last week, the first week, and the week
before the last week of the enrollment

117~120: average, standard deviation, maximal, minimal weekly numbers of
events in the enrollment period

121~122: coefficients b and c in the polynomial model y = a + bx + cx**2,
where x is week number (all start from 0), and y is the weekly number of
events

123~129: 7 counts of events in Monday to Sunday

130~153: 24 counts of events in hour 0-23

154~160: 7 counts of event types

161~162: 2 counts of source types
"""


import logging
import sys

import numpy as np
import pandas as pd

import IO
import dataset


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')


def extract(enrollment, base_date):
    enroll_all = dataset.FULL_DATASET['enroll']
    log_all = dataset.FULL_DATASET['log']
    log_all = log_all[log_all['time'] <= base_date]
    obj_all = dataset.FULL_DATASET['obj']

    return None
