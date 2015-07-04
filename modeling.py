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


def lr_with_scale():
    """
    Submission: lr_with_scale_0703_01.csv
    E_val: <missing>
    E_in: 0.878883
    E_out: 0.8766589627938616
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X_scaled, y)
    print('CV scores: %s' % clf.scores_)
    print('Ein: %f' % Util.auc_score(clf, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('lr', clf)]), 'lr_with_scale_0703_01')


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
        pl.savefig('lr_with_fs.refcv')

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


def lr_with_fs1():
    """
    Submission: lr_with_fs1_0703_03.csv
    E_val:
    E_in: 0.876954
    E_out:
    """
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    pkl_path = Path.of_cache('lr_with_fs1.LR.FS.pkl')
    lr = IO.fetch_cache(pkl_path)
    if lr is None:
        lr = LogisticRegression(class_weight='auto')
        lr.fit(X_scaled, y)
        IO.cache(lr, pkl_path)

    X_pruned = lr.transform(X_scaled)

    new_scaler = StandardScaler()
    new_scaler.fit(X_pruned)
    X_new = new_scaler.transform(X_pruned)

    clf = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    clf.fit(X_new, y)

    print('CV scores: %s' % clf.scores_)
    print('Ein: %f' % Util.auc_score(clf, X_new, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('fs', lr),
                                 ('scale_new', new_scaler),
                                 ('lr', clf)]), 'lr_with_fs1_0703_03')


def svc_test():
    """
    Submission:
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    svc = SVC(kernel='linear', class_weight='auto')
    svc.fit(X_scaled, y)

    isotonic = CalibratedClassifierCV(svc, cv=StratifiedKFold(y, 5),
                                      method='isotonic')
    isotonic.fit(X_scaled, y)

    logger.debug('Got best isotonic CalibratedClassifier.')
    logger.debug('E_in (isotonic): %f', Util.auc_score(isotonic, X_scaled, y))


def svc_test2():
    """
    Submission:
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.calibration import CalibratedClassifierCV

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    svc = SVC(kernel='linear', class_weight='auto', cache_size=10240)
    svc.fit(X_scaled, y)

    isotonic = CalibratedClassifierCV(svc, cv=StratifiedKFold(y, 5),
                                      method='isotonic')
    isotonic.fit(X_scaled, y)

    logger.debug('Got best isotonic CalibratedClassifier.')
    logger.debug('E_in (isotonic): %f', Util.auc_score(isotonic, X_scaled, y))


def svc():
    """
    Submission: svc_0703_04.csv
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.grid_search import RandomizedSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from scipy.stats import expon

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    svc = SVC(kernel='linear', class_weight='auto')
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            cv=StratifiedKFold(y, 5),
                            param_distributions={'C': expon()})
    rs.fit(X_scaled, y)

    logger.debug('Got best SVC.')
    logger.debug('Grid scores: %s', rs.grid_scores_)
    logger.debug('Best score (E_val): %s', rs.best_score_)
    logger.debug('Best params: %s', rs.best_params_)
    IO.cache(rs, Path.of_cache('svc.RandomizedSearchCV.SVC.pkl'))

    svc = rs.best_estimator_
    IO.cache(rs, Path.of_cache('svc.SVC.pkl'))

    isotonic = CalibratedClassifierCV(svc, cv=StratifiedKFold(y, 5),
                                      method='isotonic')
    isotonic.fit(X_scaled, y)
    IO.cache(rs, Path.of_cache('svc.CalibratedClassifierCV.isotonic.pkl'))

    logger.debug('Got best isotonic CalibratedClassifier.')
    logger.debug('E_in (isotonic): %f', Util.auc_score(isotonic, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('svc', isotonic)]), 'svc_0703_04')


def knn():
    """
    Submission: knn_0704_01.csv
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.grid_search import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    import numpy as np

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    knn = KNeighborsClassifier()
    params = {
        'n_neighbors': np.arange(5, 51, 5),
        'weights': ['uniform', 'distance'],
        'leaf_size': np.arange(30, 201, 10)
    }
    grid = GridSearchCV(knn, params, scoring='roc_auc', n_jobs=-1,
                        cv=StratifiedKFold(y, 5))
    grid.fit(X_scaled, y)

    logger.debug('Got best kNN.')
    logger.debug('Grid scores: %s', grid.grid_scores_)
    logger.debug('Best score (E_val): %s', grid.best_score_)
    logger.debug('Best params: %s', grid.best_params_)
    IO.cache(grid, Path.of_cache('knn.GridSearchCV.KNeighborsClassifier.pkl'))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('knn', grid)]), 'knn_0704_01')


def lr_with_scale2():
    """
    Submission: lr_with_scale2_0704_03.csv
    E_val:
    E_in: 0.878996
    E_out: 0.8768131004917349
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    clf = LogisticRegressionCV(Cs=50, cv=5, scoring='roc_auc', n_jobs=-1,
                               class_weight='auto')
    clf.fit(X_scaled, y)
    logger.debug('Best C: %f', clf.C_[0])
    logger.debug('Cs: %s', clf.Cs_)
    logger.debug('Grid scores: %f', clf.scores_)
    logger.debug('Ein: %f', Util.auc_score(clf, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('lr', clf)]), 'lr_with_scale2_0704_03')


def rf():
    """
    Submission: rf_0704_02.csv
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    rf = RandomForestClassifier(n_estimators=30000, oob_score=True, n_jobs=-1,
                                class_weight='auto')
    rf.fit(X_scaled, y)

    logger.debug('Eval(oob): %f', rf.oob_score_)
    logger.debug('Ein: %f', Util.auc_score(rf, X_scaled, y))

    IO.cache(rf, Path.of_cache('rf.RandomForestClassifier.300.pkl'))
    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('rf', rf)]), 'rf_0704_02')


if __name__ == '__main__':
    from inspect import isfunction
    variables = locals()
    if len(sys.argv) > 1:
        for fn in sys.argv[1:]:
            if fn not in variables or not isfunction(variables[fn]):
                print('function %s not found' % repr(fn))
                sys.exit(-1)
            else:
                variables[fn]()
