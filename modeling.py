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
    Submission: rf_0708_01.csv
    3000 trees
    E_val: 0.871837
    E_in: 0.999998
    E_out: 0.882316801296279
    15000 trees
    E_val: 0.872011
    E_in: 0.999998
    E_out: 0.8824869811781106
    30000 trees
    E_val: 0.871928
    E_in:
    E_out:

    depth=4, 15000 trees
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(np.r_[X, dataset.load_test()])
    X_scaled = raw_scaler.transform(X)
    del X

    rf = RandomForestClassifier(n_estimators=15000, oob_score=True, n_jobs=-1,
                                class_weight='auto')
    rf.fit(X_scaled, y)

    logger.debug('RandomForestClassifier fitted')

    import gc
    gc.collect()

    logger.debug('E_val(oob): %f', rf.oob_score_)
    logger.debug('E_in: %f', Util.auc_score(rf, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('rf', rf)]), 'rf_0708_01')

    logger.debug('caching fitted RandomForestClassifier')
    IO.cache(rf, Path.of_cache('rf.RandomForestClassifier.30000.pkl'))
    logger.debug('cached fitted RandomForestClassifier')


def rf2():
    """
    Submission: rf2_0704_04.csv
    3000 trees
    E_val: 0.871431
    E_in: 0.999998
    E_out:
    30000 trees
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
                                class_weight='auto', max_features='log2')
    rf.fit(X_scaled, y)

    logger.debug('Eval(oob): %f', rf.oob_score_)
    logger.debug('Ein: %f', Util.auc_score(rf, X_scaled, y))

    IO.cache(rf, Path.of_cache('rf.RandomForestClassifier.log2.pkl'))
    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('rf', rf)]), 'rf2_0704_04')


def erf():
    """
    Submission: erf_0705_01.csv
    3000 trees
    E_val: 0.870800
    E_in: 0.999998
    E_out:
    15000 trees
    E_val:
    E_in:
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)
    del X

    rf = ExtraTreesClassifier(n_estimators=3000, oob_score=True, n_jobs=-1,
                              class_weight='auto', bootstrap=True)
    rf.fit(X_scaled, y)

    logger.debug('ExtraTreesClassifier fitted')

    import gc
    gc.collect()

    logger.debug('Eval(oob): %f', rf.oob_score_)
    logger.debug('Ein: %f', Util.auc_score(rf, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('erf', rf)]), 'erf_0705_01')

    logger.debug('caching fitted ExtraTreesClassifier')
    IO.cache(rf, Path.of_cache('erf.ExtraTreesClassifier.auto.pkl'))
    logger.debug('cached fitted ExtraTreesClassifier')


def erf2():
    """
    Submission: erf2_0705_02.csv
    3000 trees
    E_val: [0.83766072, 0.89704662, 0.85299486, 0.8639041, 0.82955865]
    E_in: 1.000000
    E_out:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.cross_validation import cross_val_score

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    del X
    import gc
    gc.collect()

    erf = ExtraTreesClassifier(n_estimators=3000, n_jobs=-1,
                               class_weight='auto')
    scores = cross_val_score(erf, X_scaled, y, cv=5, n_jobs=-1)
    logger.debug('CV: %s', scores)
    logger.debug('Eval: %f', sum(scores) / len(scores))

    erf.fit(X_scaled, y)
    logger.debug('ExtraTreesClassifier fitted')
    logger.debug('Ein: %f', Util.auc_score(erf, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('erf', erf)]), 'erf2_0705_02')

    logger.debug('caching fitted ExtraTreesClassifier')
    IO.cache(erf, Path.of_cache('erf2.ExtraTreesClassifier.auto.pkl'))
    logger.debug('cached fitted ExtraTreesClassifier')


def gbdt_oob():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import KFold, train_test_split
    import pylab as pl
    import numpy as np

    X, y = dataset.load_train()
    raw_scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    raw_scaler.fit(X_train)
    X_train = raw_scaler.transform(X_train)
    X_test = raw_scaler.transform(X_test)

    n_estimators = 1000
    params = {'n_estimators': n_estimators, 'loss': 'deviance',
              'learning_rate': 0.1, 'subsample': 0.5}
    gb = GradientBoostingClassifier(**params)

    gb.fit(X_train, y_train)

    # IO.cache(gb, Path.of_cache('gbdt.GradientBoostingClassifier.pkl'))

    logger.debug('Eval: %f', Util.auc_score(gb, X_test, y_test))
    logger.debug('Ein: %f', Util.auc_score(gb, X_train, y_train))

    x = np.arange(n_estimators) + 1

    def heldout_score(clf, X_test, y_test):
        """compute deviance scores on ``X_test`` and ``y_test``. """
        score = np.zeros((n_estimators,), dtype=np.float64)
        for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
            score[i] = clf.loss_(y_test, y_pred)
        return score

    def cv_estimate(n_folds=3):
        cv = KFold(n=X_train.shape[0], n_folds=n_folds)
        cv_clf = GradientBoostingClassifier(**params)
        val_scores = np.zeros((n_estimators,), dtype=np.float64)
        for train, test in cv:
            cv_clf.fit(X_train[train], y_train[train])
            val_scores += heldout_score(cv_clf, X_train[test], y_train[test])
        val_scores /= n_folds
        return val_scores

    cv_score = cv_estimate(3)
    test_score = heldout_score(gb, X_test, y_test)
    cumsum = -np.cumsum(gb.oob_improvement_)
    oob_best_iter = x[np.argmin(cumsum)]
    test_score -= test_score[0]
    test_best_iter = x[np.argmin(test_score)]
    cv_score -= cv_score[0]
    cv_best_iter = x[np.argmin(cv_score)]

    oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
    test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
    cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

    # IO.cache(cumsum, Path.of_cache('gbdt.cumsum.pkl'))
    # IO.cache(test_score, Path.of_cache('gbdt.test_score.pkl'))
    # IO.cache(cv_score, Path.of_cache('gbdt.cv_score.pkl'))

    pl.plot(x, cumsum, label='OOB loss', color=oob_color)
    pl.plot(x, test_score, label='Test loss', color=test_color)
    pl.plot(x, cv_score, label='CV loss', color=cv_color)
    pl.axvline(x=oob_best_iter, color=oob_color)
    pl.axvline(x=test_best_iter, color=test_color)
    pl.axvline(x=cv_best_iter, color=cv_color)

    xticks = pl.xticks()
    xticks_pos = np.array(xticks[0].tolist() +
                          [oob_best_iter, cv_best_iter, test_best_iter])
    xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +
                            ['OOB', 'CV', 'Test'])
    ind = np.argsort(xticks_pos)
    xticks_pos = xticks_pos[ind]
    xticks_label = xticks_label[ind]
    pl.xticks(xticks_pos, xticks_label)

    pl.legend(loc='upper right')
    pl.ylabel('normalized loss')
    pl.xlabel('number of iterations')

    pl.savefig('gbdt.oob')


def gbdt():
    """
    Submission: gbdt_0706_01.csv
    n_estimators: 1000, learning_rate: 0.1, subsample: 0.5
    E_val: 0.858235
    E_in: 0.908622
    E_out: 0.8873906795559863
    n_estimators: 500, learning_rate: 0.1, subsample: 0.5
    E_val: 0.870976
    E_in: 0.899593
    E_out: 0.88711101837711
    n_estimators: 3000, learning_rate: 0.1, subsample: 0.5
    E_val: 0.836049
    E_in: 0.936056
    E_out: 0.8833930861722906
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import cross_val_score

    X, y = dataset.load_train()
    clf = Pipeline([('scaler', StandardScaler()),
                    ('gbdt', GradientBoostingClassifier(
                        n_estimators=3000, learning_rate=0.1, subsample=0.5))])

    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc', n_jobs=-1,
                             verbose=1)

    logger.debug('E_val: %f <- %s', sum(scores) / len(scores), scores)

    clf.fit(X, y)
    IO.cache(clf, Path.of_cache('gbdt.Pipeline.pkl'))
    logger.debug('E_in: %f', Util.auc_score(clf, X, y))

    IO.dump_submission(clf, 'gbdt_0706_01')


def gbdt2():
    """
    Submission: gbdt2_0706_02.csv
    n_estimators: 1000, learning_rate: 0.1, subsample: 0.5
    E_val: 0.852035
    E_in: 0.910251
    E_out: 0.8874428893001793
    n_estimators: 3000, learning_rate: 0.1, subsample: 0.5
    E_val: 0.827988
    E_in: 0.938593
    E_out: 0.8844206314551558
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import cross_val_score

    X, y = dataset.load_train()
    clf = Pipeline([('scaler', StandardScaler()),
                    ('gbdt', GradientBoostingClassifier(
                        loss='exponential', n_estimators=3000,
                        learning_rate=0.1, subsample=0.5))])

    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc', n_jobs=-1,
                             verbose=1)
    logger.debug('E_val: %f <- %s', sum(scores) / len(scores), scores)

    clf.fit(X, y)
    IO.cache(clf, Path.of_cache('gbdt2.Pipeline.pkl'))
    logger.debug('E_in: %f', Util.auc_score(clf, X, y))

    IO.dump_submission(clf, 'gbdt2_0706_02')


def gbdt_grid():
    """
    Grid search for best params.
    Best params: {'learning_rate': 0.05, 'subsample': 0.3}
    Submission: gbdt_grid_0706_03.csv
    E_val: 0.860118290628
    E_in: 0.882949
    E_out: 0.8809314555068068
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.pipeline import Pipeline
    import numpy as np

    X, y = dataset.load_train()
    raw_scaler = StandardScaler()
    X_scaled = raw_scaler.fit_transform(X)

    param_grid = {
        'learning_rate': [0.05, 0.1],
        'subsample': [0.3, 0.5, 0.7]
    }

    grid = GridSearchCV(GradientBoostingClassifier(n_estimators=3000),
                        param_grid, scoring='roc_auc', n_jobs=-1,
                        cv=StratifiedKFold(y, 5), refit=False, verbose=1)
    grid.fit(X_scaled, y)

    logger.debug('Got best GBDT.')
    logger.debug('Grid scores: %s', grid.grid_scores_)
    logger.debug('Best score (E_val): %s', grid.best_score_)
    logger.debug('Best params: %s', grid.best_params_)

    X_test = dataset.load_test()
    raw_scaler.fit_transform(np.r_[X, X_test])
    X_scaled = raw_scaler.transform(X)

    clf = GradientBoostingClassifier(**grid.best_params_)
    clf.fit(X_scaled, y)

    IO.cache(grid, Path.of_cache('gbdt_grid.GridSearchCV.pkl'))

    logger.debug('E_in: %f', Util.auc_score(clf, X_scaled, y))
    IO.dump_submission(Pipeline([('scaler', raw_scaler),
                                 ('gbdt', clf)]), 'gbdt_grid_0706_03')


def gbdt_search():
    """
    Grid search for best n_estimators.
    Best params: {'loss': 'deviance', 'n_estimators': 100}
    Submission: gbdt_search_0707_01.csv
    E_val: 0.883786743214
    E_in: 0.887785
    E_out: 0.8848760405053878
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.grid_search import GridSearchCV
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.pipeline import Pipeline
    import numpy as np

    X, y = dataset.load_train()
    raw_scaler = StandardScaler()
    X_scaled = raw_scaler.fit_transform(X)

    param_grid = {
        'loss': ['deviance', 'exponential'],
        'n_estimators': np.arange(100, 1001, 100)
    }

    params = {'learning_rate': 0.1, 'subsample': 0.5}

    gb = GradientBoostingClassifier(**params)
    grid = GridSearchCV(gb, param_grid, scoring='roc_auc', n_jobs=-1,
                        cv=StratifiedKFold(y, 5), refit=True, verbose=1)
    grid.fit(X_scaled, y)

    logger.debug('Got best GBDT.')
    logger.debug('Grid scores: ')
    for i, grid_score in enumerate(grid.grid_scores_):
        print('\t%d00: %s' % (i + 1, grid_score))
    logger.debug('Best score (E_val): %s', grid.best_score_)
    logger.debug('Best params: %s', grid.best_params_)

    IO.cache(grid, Path.of_cache('gbdt_search.GridSearchCV.pkl'))

    X_test = dataset.load_test()
    raw_scaler.fit(np.r_[X, X_test])
    X_scaled = raw_scaler.transform(X)

    params.update(grid.best_params_)
    clf = GradientBoostingClassifier(**params)
    clf.fit(X_scaled, y)

    logger.debug('E_in: %f', Util.auc_score(grid, X_scaled, y))
    IO.dump_submission(Pipeline([('scaler', raw_scaler),
                                 ('gbdt', grid)]), 'gbdt_search_0707_01')


def bagging_lr():
    """
    Submission: bagging_lr_0707_02.csv
    E_val:
    E_in:
    E_out:
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import BaggingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    bag = BaggingClassifier(LogisticRegression(class_weight='auto'),
                            n_estimators=3000, oob_score=True, n_jobs=-1,
                            verbose=2)

    logger.debug('E_val (oob): %f', bag.oob_score_)
    logger.debug('E_in: %f', Util.auc_score(bag, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('bag', bag)]), 'bagging_lr_0707_02')


def ada_boost_dt():
    """
    Submission: ada_boost_dt_0707_03.csv
    E_val: 0.854350
    E_in: 0.889561
    E_out: 0.8832315976033993
    """
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import cross_val_score
    from sklearn.pipeline import Pipeline

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    ab = AdaBoostClassifier(n_estimators=300)

    scores = cross_val_score(ab, X_scaled, y, cv=5, n_jobs=-1)
    logger.debug('CV: %s', scores)
    logger.debug('E_val: %f', sum(scores) / len(scores))

    ab.fit(X_scaled, y)

    logger.debug('E_in: %f', Util.auc_score(ab, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('ab', ab)]), 'ada_boost_dt_0707_03')


def svc_appr():
    """
    Best params: {'C': 0.022139881953014046}

    Submission:
    E_val:
    E_in:
    E_out:
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.grid_search import RandomizedSearchCV
    from scipy.stats import expon

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    svc = LinearSVC(dual=False, class_weight='auto')
    rs = RandomizedSearchCV(svc, n_iter=50, scoring='roc_auc', n_jobs=-1,
                            cv=StratifiedKFold(y, 5), verbose=2,
                            param_distributions={'C': expon()})
    rs.fit(X_scaled, y)

    logger.debug('Got best SVC.')
    logger.debug('Best params: %s', rs.best_params_)
    logger.debug('Grid scores:')
    for i, grid_score in enumerate(rs.grid_scores_):
        print('\t%s' % grid_score)
    logger.debug('Best score (E_val): %s', rs.best_score_)
    logger.debug('E_in: %f', Util.auc_score(rs, X_scaled, y))


def lr_with_scale3():
    """
    Check the performance of normalizing TEST SET.

    Submission: lr_with_scale3_0707_04.csv
    E_val:
    E_in: 0.879233
    E_out: 0.8770121701777971
    """
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    X, y = dataset.load_train()

    raw_scaler = StandardScaler()
    raw_scaler.fit(X)
    X_scaled = raw_scaler.transform(X)

    clf = LogisticRegressionCV(Cs=50, cv=5, scoring='roc_auc', n_jobs=-1,
                               class_weight='auto', refit=False)
    clf.fit(X_scaled, y)
    logger.debug('Best C: %f', clf.C_[0])
    logger.debug('Cs: %s', clf.Cs_)
    logger.debug('Grid scores: %f', clf.scores_)

    clf = LogisticRegression(C=clf.C_[0], class_weight='auto')

    raw_scaler.fit(np.r_[X, dataset.load_test()])
    X_scaled = raw_scaler.transform(X)

    clf.fit(X_scaled, y)

    logger.debug('E_in: %f', Util.auc_score(clf, X_scaled, y))

    IO.dump_submission(Pipeline([('scale_raw', raw_scaler),
                                 ('lr', clf)]), 'lr_with_scale3_0707_04')


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
