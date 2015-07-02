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

import numpy as np
import pandas as pd

import IO


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger(os.path.basename(__file__))


def extract(base_date):
    logger.debug('prepare datasets ...')

    enroll_all = IO.load_enrollments()
    log_all = IO.load_logs()
    obj_all = IO.load_object()

    log_all = log_all[log_all['time'] <= base_date]

    logger.debug('datasets prepared')

    return None
