#! /usr/local/bin/python3
# -*- utf-8 -*-


import numpy as np
import pandas as pd


def dataframe_checker(logger):
    def check(df, msg):
        nr, nc = df.shape
        logger.debug('%s: shape: %d x %d', msg, nr, nc - 1)
        logger.debug('%s: columns: %s', msg, df.columns.values)
        if np.any(pd.isnull(df)):
            for c in df.columns.values:
                if np.any(pd.isnull(df[c])):
                    logger.debug('%s: column `%s` has Na', msg, c)
    return check


def auc_score(clf, X, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, clf.predict_proba(X)[:, 1])
