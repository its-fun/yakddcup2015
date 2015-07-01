#! /usr/local/bin/python3
# -*- utf-8 -*-


import numpy as np
import pandas as pd


def dataframe_checker(logger):
    def check(df, msg):
        nr, nc = df.shape
        logger.debug('%s: columns: %s', msg, df.columns.values)
        logger.debug('%s: shape: %d x %d, has_nan: %s', msg, nr, nc - 1,
                     np.any(pd.isnull(df)))
        if np.any(pd.isnull(df)):
            for c in df.columns.values:
                if np.any(pd.isnull(df[c])):
                    logger.debug('%s: column `%s` has Na', msg, c)
    return check
