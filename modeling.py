#! /usr/local/bin/python3
# -*- utf-8 -*-


import logging
import sys

import dataset
import IO
import Path
import Util

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s\t%(message)s')
logger = logging.getLogger('modeling')


def lr_with_fs():
    """
    Submission: lr_with_fs_0703_01.csv
    E_val:
    E_in:
    E_out:
    """
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    import pylab as pl

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    pkl_path = Path.of_cache('lr_with_fs.RFECV.pkl')
    rfe = IO.fetch_cache(pkl_path)
    if rfe is None:
        rfe = RFECV(estimator=LogisticRegression(class_weight='auto'),
                    cv=StratifiedKFold(y, 5), scoring='roc_auc')
        rfe.fit(X_scaled, y)
        IO.cache(rfe, pkl_path)

        print("Optimal number of features : %d" % rfe.n_features_)

        # Plot number of features VS. cross-validation scores
        pl.figure()
        pl.xlabel("Number of features selected")
        pl.ylabel("Cross validation score (AUC)")
        pl.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
        pl.show()

    X_pruned = rfe.transform(X_scaled)

    new_scaler = StandardScaler()
    new_scaler.fit(X_pruned)
    X_new = new_scaler.transform(X_pruned)

    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X_new, y)

    print('CV scores: %s' % clf.scores_)
    print('Ein: %f' % Util.auc_score(clf, X_new, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('rfe', rfe),
                                 ('scale_new', new_scaler),
                                 ('lr', clf)]), 'lr_with_fs_0703_01')


if __name__ == '__main__':
    from inspect import isfunction
    variables = locals()
    if len(sys.argv) > 1:
        for fn in sys.argv[1:]:
            if fn not in variables or not isfunction(variables[fn]):
                print('function %s not found' % repr(fn))
            variables[fn]()
