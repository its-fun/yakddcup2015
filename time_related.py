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

import numpy as np
import pandas as pd

import fio
import dataset


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')


def extract(enrollment, base_date):
    enroll_all = dataset.FULL_DATASET['enroll']
    log_all = dataset.FULL_DATASET['log']
    log_all = log_all[log_all['time'] <= base_date]
    obj_all = dataset.FULL_DATASET['obj']

    return None
