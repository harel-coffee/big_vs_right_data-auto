import pandas as pd
import numpy as np

from scipy.stats import randint as sp_randint
from scipy.stats import lognorm as sp_lognormal
from scipy.stats import uniform as sp_uniform

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer    
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, \
    RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel



model_error = ValueError("Model not implemented!")

def predict_class(input_tuple,
                  weighted=True):

    X_, y_, features, split, model = input_tuple
    

    feature_set_label, feature_set = features    
    X_arr = X_[feature_set].values.copy()
    y_arr = y_.values.copy()
    all_index = y_.index.values
    k = len(feature_set)
    
    
    split_number, train_idx, test_idx = split
    X_train, X_test = X_arr[train_idx], X_arr[test_idx]
    y_train, y_test = y_arr[train_idx], y_arr[test_idx]

    
    # baseline estimate (choose majority)
    if len(feature_set) == 0:
        majority_class = y_.value_counts().index[0]
        y_pred = [majority_class]*len(test_idx)
        params_opt = None

    else:
        if model['estimate_hyper_param']:
            
            # estimate models with hyperparameter tuning
            def _make_clf():
                if model['clf'] == "RF":
                    _clf = RandomForestClassifier
                    _params = dict(n_estimators=100, 
                                   n_jobs=1)
                    grid_params = dict(max_depth = sp_randint(4,50),
                                       min_samples_leaf = sp_randint(1,7),
                                       min_samples_split = sp_randint(2,10))

                elif model['clf'] == "LR":
                    _clf = LogisticRegression
                    _params = dict(multi_class='auto',
                                   solver='liblinear')
                    grid_params = dict(C = sp_lognormal(scale=1,s=3))

                else:
                    raise model_error 

                _params.update({**dict(class_weight='balanced'),
                                **model['clf_params']})

                return _clf(random_state=0, **_params), grid_params

            clf_steps = [('impute', SimpleImputer(strategy='mean'))]

            grid_params = {}

            # add polynomial features
            if model['poly_degree']>1:
                clf_steps += [('poly_feat', PolynomialFeatures(degree=model['poly_degree'],include_bias=False))]

            # add feature selection step        
            feature_count_random = [1] if k==1 else sp_randint(1, k)       
            if model['fs']:
                clf, clf_grid = _make_clf()
                sfm = SelectFromModel(estimator=clf, 
                                      threshold=-np.inf)
                clf_steps += [('fs', sfm)]

                for k,v in clf_grid.items():
                    grid_params['fs__estimator__'+k] = v
                grid_params['fs__max_features'] = feature_count_random


            # add model estimation step
            clf, clf_grid = _make_clf()
            clf_steps += [('clf', clf)]
            for k,v in clf_grid.items():
                grid_params['clf__'+k] = v

            clf_pipe = Pipeline(steps=clf_steps)

            # normalize data
            if model != 'RF':
                sc = StandardScaler()
                sc.fit(X_train)
                X_train = sc.transform(X_train)
                X_test = sc.transform(X_test)


            search = RandomizedSearchCV(clf_pipe,
                                        grid_params,
                                        cv=RepeatedStratifiedKFold(5,2),
                                        iid=False,
                                        n_iter=100,
                                        n_jobs=1)

            search.fit(X_train, y_train)        
            params_opt = search.best_params_

            y_pred = search.best_estimator_.predict(X_test)
            
        else:
            # estimate models without hyperparameter tuning
            params_opt = model['clf_params']
            clf = \
                Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 
                                ('clf', RandomForestClassifier(n_estimators=100, 
                                                               random_state=0, 
                                                               **model['clf_params']))]
                        )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        
    if weighted:    
        w_test = y_.iloc[test_idx].map(1/y_.iloc[test_idx].value_counts()).values
        acc = accuracy_score(y_test, y_pred, sample_weight=w_test)

    else:
        acc = accuracy_score(y_test, y_pred)

    return feature_set_label, split_number, acc, y_pred, params_opt