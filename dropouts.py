#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Total: 11

0~1: 课程最后十天无操作的人数，与选课人数的比例

2~5: 用户平均每个课程Dropout的次数，在该课程中Dropout的次数，与平均每个课程Dropout次数的比例；
总Dropout持续时长与课程持续时间的比例

6~10: 用户在所有课程上Dropout总持续时长与课程持续时间的比例的：平均值、方差、最大值、最小值；
有Dropout行为的课程占用户所选课程总数的比例
"""


import logging
import sys

import numpy as np
import pandas as pd

import IO
import dataset


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(__file__)


def extract(enrollment, base_date):
    enroll_all = dataset.FULL_DATASET['enroll']
    log_all = dataset.FULL_DATASET['log']
    log_all = log_all[log_all['time'] <= base_date]
    obj_all = dataset.FULL_DATASET['obj']

    return None
