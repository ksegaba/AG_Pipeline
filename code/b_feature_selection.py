'''
Train a Random Forest classifier to perform feature selection using Gini
impurity or permutation importances. Alternatively, calculate average absolute
SHAP values from a pre-trained decision tree-based model (from AutoGluon or
Scikit-learn) and generate feature files.

Arguments:
    # required arguments
    -X: Feature table
    -output_dir: Directory to save the results to

    # optional arguments
    -fstype: Feature selection importance measure type(shap/permutation/gini)[default: shap]
    -model_path: Path to a pre-trained model for calculating SHAP values[default: '']
    -Y: Label file if label not in X [default: '']
    -y_name: Name of the label column in X [default: 'Class']
    -test: Test instances file [default: '']
    -start: Start number of features
    -stop: Stop number of features
    -step: Step number of features
    -prefix: Prefix for the output files
    -write: Write the selected features to a file(y/n)[default: y]

Output:
- Fitted Random Forest model(_fs_model.joblib)
- Feature importances[gini and permutation](_fs_importance.csv)
- Selected features at each step(_fs_feats_permutation.json or _fs_feats_gini.json)
'''

__author__ = 'Kenia Segura Abá'

import sys
import os
import argparse
import time
import json
import joblib
import shap
import pandas as pd
import datatable as dt
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def parse_args():
    parser = argparse.ArgumentParser(description='Feature selection')
    # required arguments
    parser.add_argument('-X', required=True, type=str, help='Feature table')
    parser.add_argument('-output_dir', required=True, type=str,
                        help='Directory to save the results to')
    #
    # optional arguments
    parser.add_argument('-fstype', default='shap', type=str,
                        help='Feature selection importance measure type (shap/permutation/gini)')
    parser.add_argument('-model_path', default='', type=str,
                        help='Path to a pre-trained model for SHAP values')
    parser.add_argument('-Y', default='', type=str,
                        help='Label file if label not in X')
    parser.add_argument('-y_name', default='Class', type=str,
                        help='Name of the label column in X')
    parser.add_argument('-test', default='', type=str,
                        help='Test instances file')
    parser.add_argument('-start', type=int, help='Start number of features')
    parser.add_argument('-stop', type=int, help='Stop number of features')
    parser.add_argument('-step', type=int, help='Step number of features')
    parser.add_argument('-prefix', default='', type=str,
                        help='Prefix for the output files')
    parser.add_argument('-write', default='y', type=str,
                        help='Write the selected features to a file (y/n)')
    args = parser.parse_args()
    return args


def f1_score_safe(y_true, y_pred):
    '''Calculate the F1 score with zero division handling
    It resolves the following error:
    UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to
    no true nor predicted samples. Use `zero_division` parameter to control this
    behavior.'''
    return f1_score(y_true, y_pred, zero_division=1)


def hyperopt_objective_clf(params, X_train_norm, y_train):
    '''
    Create the hyperparameter grid and run Hyperopt hyperparameter tuning
    with K-fold cross-validation for RandomForestClassifier
    Written by Thejesh Mallidi
    Modified by Kenia Segura Abá
    '''

    mod = RandomForestClassifier(**params, random_state=321)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score_safe)
    validation_loss = cross_validate(
        mod, X_train_norm, y_train,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        error_score='raise'
    )

    # Note: Hyperopt minimizes the objective, so we want to minimize the loss, thereby maximizing the F1 score
    loss = -np.mean(validation_loss['test_score'])
    return {'loss': loss, 'status': STATUS_OK}


def param_hyperopt(param_grid, X_train_norm, y_train, max_evals=100, type='c'):
    '''
    Obtain the best parameters from Hyperopt
    Written by Thejesh Mallidi
    '''
    trials = Trials()

    if type == 'c':
        params_best = fmin(
            fn=lambda params: hyperopt_objective_clf(
                params, X_train_norm, y_train),
            space=param_grid,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=1
        )

    print('\n\nBest parameters:', params_best)
    return params_best, trials


def feature_selection_clf(X_train, y_train, start, stop, step, output_dir, prefix, write='n', type='permutation'):
    ################ Build the initial model using all features ################
    # Hyperparameter tuning
    X_train_norm = MinMaxScaler().fit_transform(X_train)  # Normalize
    X_train_norm = pd.DataFrame(
        X_train_norm, columns=X_train.columns, index=X_train.index)
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 5, 400, 5)),
        'max_depth': scope.int(hp.quniform('max_depth', 2, 6, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'max_features': hp.choice('max_features', ['sqrt', 'log2'])
    }
    best_params, trials = param_hyperopt(
        space, X_train_norm, y_train, max_evals=200, type='c')
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    best_params['max_features'] = ['sqrt', 'log2'][best_params['max_features']]

    # Fit the model with the best parameters
    forest = RandomForestClassifier(**best_params, random_state=321)
    forest.fit(X_train_norm, y_train)
    with open(f'{output_dir}/{prefix}_fs_model.joblib', 'wb') as f:
        joblib.dump(forest, f)

    # Feature permutation importance
    gini = pd.DataFrame(
        forest.feature_importances_, index=X_train.columns, columns=['gini'])

    print('Calculating permutation importance...')
    result = permutation_importance(
        forest, X_train_norm, y_train, n_repeats=10, random_state=321, n_jobs=2)

    importances = pd.concat([pd.DataFrame(result['importances'], index=X_train.columns),
                             pd.DataFrame(
                                 result['importances_mean'], index=X_train.columns, columns=['mean']),
                             pd.DataFrame(result['importances_std'], index=X_train.columns, columns=['std'])],
                            axis=1)
    importances.sort_values(by='mean', ascending=False, inplace=True)
    importances = pd.concat([importances, gini], axis=1, ignore_index=False)
    importances.to_csv(f'{output_dir}/{prefix}_fs_importance.csv')

    # Calculate Feature permutation importance within K-fold cross-validation
    # start_time = time.time()
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=321)
    # importances = []

    # for i, (train_idx, val_idx) in enumerate(cv.split(X_train_norm, y_train)):
    # 	# Train-validation split
    # 	X_train_cv, X_val = X_train_norm.iloc[train_idx], X_train_norm.iloc[val_idx]
    # 	y_train_cv, y_val = y_train[train_idx], y_train[val_idx]

    # 	# Fit the model
    # 	forest.fit(X_train_cv, y_train_cv)
    # 	result = permutation_importance(
    # 		forest, X_val, y_val, n_repeats=10, random_state=i, n_jobs=2)
    # 	importances.append(result.importances_mean)

    # end_time = time.time()
    # print(f'Permutation importance time: {end_time - start_time}')

    # importances = pd.DataFrame(np.array(importances), columns=X_train.columns)
    # imp_mean = np.array(importances).mean(axis=0)
    # imp_sd = np.array(importances).std(axis=0)

    ####################### Iterative feature selection ########################
    # Select features based on the permutation importance
    if type == 'permutation':
        selected_features = []
        for t in range(start, stop, step):
            selected = importances.index[:t].to_list()
            selected_features.append(selected)

        if write == 'y':
            json.dump(selected_features,
                      open(f'{output_dir}/{prefix}_fs_feats_permutation.json', 'w'), indent=2)

    if type == 'gini':
        gini.sort_values(by='gini', ascending=False, inplace=True)
        selected_features = []
        for t in range(start, stop, step):
            selected = gini.index[:t].to_list()
            selected_features.append(selected)

        if write == 'y':
            json.dump(selected_features,
                      open(f'{output_dir}/{prefix}_fs_feats_gini.json', 'w'), indent=2)

    print(f'{len(selected_features)} sets of selected features from {start} to {stop} with step {step}')

    return selected_features


def calculate_shap_values(model_path, X_array, X):
    # 1) Load a pre-trained decision tree-based model
    model = joblib.load(model_path)  # Use joblib to load the model
    #
    # 2) Calculate SHAP values for each feature for the positive class
    try:
        explainer = shap.TreeExplainer(model)
    except:  # for AutoGluon models
        explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_array)  # Positive class is at idx 1
    shap_values = pd.DataFrame(
        shap_values[:, :, 1], columns=X.columns, index=X.index)
    return shap_values


def generate_feature_sets(shap_values, output_dir, start=None, stop=None, step=None):
    '''3) Generate feature sets for running feature selection
    Inputs:
      - shap_values: SHAP values for each feature for the positive class
      - output_dir: directory to save the feature sets to
      - start: start number of features(optional)
      - stop: stop number of features(optional)
      - step: step number of features(optional)
    Outputs:
      - feature_set_top_{n}.txt: list of top n features for each n if
        start, stop, and step are provided
      - feature_set_above_p{percentile}.txt: list of features above the
        threshold for each percentile if the threshold is unique. Only made if
        start, stop, and step are not provided.
    '''
    #
    # Average absolute SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values),
                            axis=0).sort_values(ascending=False)
    mean_abs_shap = mean_abs_shap[mean_abs_shap > 0]  # include only non-zero
    #
    # Determine unique thresholds for feature selection
    unique_thresholds = set()
    feature_sets = {}

    if start and stop and step:
        for n in range(start, stop, step):
            # Select the top n features
            selected_features = mean_abs_shap.index[:n]
            feature_sets[n] = selected_features
            #
            # Save the feature set to a file
            feature_set_path = os.path.join(
                output_dir, f'feature_set_top_{n}.txt')
            pd.DataFrame(selected_features).to_csv(
                feature_set_path, index=False, header=False, quoting=1,
                quotechar='"')
    else:
        percentiles = np.arange(0.1, 1.0, 0.1)
        #
        for percentile in percentiles:
            threshold = mean_abs_shap.quantile(round(percentile, 1))
            #
            if threshold not in unique_thresholds:
                unique_thresholds.add(threshold)
                #
                # Select features above the threshold
                feature_sets[round(percentile, 1)] = mean_abs_shap[
                    mean_abs_shap >= threshold].index
                #
                # Save the feature set to a file
                feature_set_path = os.path.join(
                    output_dir, f'feature_set_above_p{round(percentile, 1)}.txt')
                pd.DataFrame(feature_sets[round(percentile, 1)]).to_csv(
                    feature_set_path, index=False, header=False, quoting=1,
                    quotechar='"')
                #
    return feature_sets


if __name__ == '__main__':
    args = parse_args()

    print("Read in data")
    X = dt.fread(args.X).to_pandas()  # feature table
    X.set_index(X.columns[0], inplace=True)

    print("Get the label from X or Y files")
    if args.Y == '':
        y = X.loc[:, args.y_name]
        X.drop(columns=args.y_name, inplace=True)
    else:
        Y = pd.read_csv(args.Y, index_col=0)
        y = Y.loc[:, args.y_name]

    y = y.astype(int)  # convert binary bool values to integer

    if args.test != '':
        print("Train-test split")
        test = pd.read_csv(args.test, header=None)  # test instances

        X_train = X.loc[~X.index.isin(test[0])]
        X_test = X.loc[test[0]]
        y_train = y.loc[~y.index.isin(test[0])]
        y_test = y.loc[test[0]]

        # Ensure rows are in the same order
        X_train = X_train.loc[y_train.index, :]
        X_test = X_test.loc[y_test.index, :]
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        print(y_train.value_counts(), y_test.value_counts())

    # Feature selection
    if args.model_path == '':
        print("Calculate either permutation or gini feature importances")
        if args.test != '':
            feature_selection_clf(X_train, y_train, args.start, args.stop, args.step,
                                  args.output_dir, args.prefix, args.write, args.fstype)
        else:
            feature_selection_clf(X, y, args.start, args.stop, args.step,
                                  args.output_dir, args.prefix, args.write, args.fstype)
    else:
        print("Calculate SHAP values")
        if args.test != '':
            shap_values = calculate_shap_values(args.model_path, X_train.to_numpy(),
                                                X_train)
        else:
            shap_values = calculate_shap_values(
                args.model_path, X.to_numpy(), X)

        # Generate feature sets
        feature_sets = generate_feature_sets(shap_values, args.output_dir)
