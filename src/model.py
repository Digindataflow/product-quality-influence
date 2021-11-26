""" Functions for training and model selection"""
import os

import numpy as np
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR


CROSS_FOLDS = 5
TOP_FEATURES = 5
RANDOM_SEED = 0
SCORING_METRICS = ["neg_mean_squared_error", "r2"]
PCA_VARIANCE = 0.95

elasticnet_file_name = "elasticnet.joblib"
svm_file_name = "svm.joblib"
rf_file_name = "rf.joblib"
gbm_file_name = "gbm.joblib"

ELASTICNET_GRID = {
    'alpha': np.logspace(-5, 2, num=8),
    'l1_ratio': np.arange(0, 1, 0.01)
}
SVR_GRID = {
    'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),
    'C' : np.logspace(-1, 2, num=4),
    'degree' : [2, 3, 4],
    'coef0' : np.linspace(start=0.01, stop=0.5, num=3),
    'gamma' : ('auto','scale')
}
GBM_GRID = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
    'colsample_bytree': [0.7, 0.8],
    'max_depth': [3, 6, 9],
    'num_leaves': [5, 10, 20],
    'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1.1, 1.2, 1.3],
    'min_gain_to_split': [0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}
RANDOMFOREST_GRID = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
    'max_depth': [int(x) for x in np.linspace(5, 15, num=6)],
    'max_leaf_nodes': [5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True, False],
}

def train_test_split_by_date(X, y, date_threshold):
    train_X, test_X = X.loc[:date_threshold, :], X.loc[date_threshold:, :]
    train_y, test_y = y.loc[:date_threshold, :], y.loc[date_threshold:, :]
    return train_X, test_X, train_y, test_y

def create_pipeline(estimator, param_grid: dict, standardize=True, pca=False):
    """create pipeline for training with cross validation
    and hyperparameter tuning

    :param estimator: Estimator：regression model
    :param param_grid: dict: hyperparameter range
    :param standardize: bool: if apply standardization
    :param pca: bool: if apply pca
    :return: Pipeline
    """
    searchcv = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_jobs=-1,
            cv=CROSS_FOLDS,
            scoring=SCORING_METRICS,
            refit=SCORING_METRICS[0]
    )
    pipelines = []
    if standardize:
        pipelines.append(preprocessing.StandardScaler())
    if pca:
        pipelines.append(PCA(n_components=PCA_VARIANCE, svd_solver="full")),
    pipelines.append(searchcv)

    model = make_pipeline(*pipelines)
    return model

def train_and_save(X, y, model, model_path):
    """train model and save it

    :param X: DataFrame: train features
    :param y: Series: train target variable
    :param model: Pipeline
    :param model_path: str: folder to save model
    :return: Pipeline: trained model
    """
    model.fit(X, y)
    dump(model, model_path)
    return model

def get_score_record(X, y, model):
    """scoring model with test data

    :param X: DataFrame: test features
    :param y: Series: test target variable
    :param model: Pipeline: trained model
    :return: dict: test score and train score
    """
    return {
        "test_mse": model.score(X, y),
        "train_mse": model.named_steps["randomizedsearchcv"].best_score_
    }

def select_model(train_X, train_y, test_X, test_y, root_path, standardize=True, pca=False):
    """train four models and record the best model result

    :param train_X: DataFrame: features
    :param train_y: Series: train target variable
    :param test_X: DataFrame: test features
    :param test_y: Series: test target variable
    :param root_path: str: folder to save model
    :param standardize: bool: if apply standardization
    :param pca: bool: if apply pca
    :return: dict: test score and train score
    """
    # create four model pipelines
    models = [
        (ElasticNet(random_state=RANDOM_SEED), ELASTICNET_GRID),
        (SVR(verbose=1), SVR_GRID),
        (RandomForestRegressor(random_state=RANDOM_SEED), RANDOMFOREST_GRID),
        (LGBMRegressor(random_state=RANDOM_SEED), GBM_GRID),
    ]
    model_paths = [elasticnet_file_name, svm_file_name, rf_file_name, gbm_file_name]
    scores = {}

    for (model_components, model_name) in zip(models, model_paths):
        trained_model = create_pipeline(*model_components, standardize=standardize, pca=pca)
        model_path = os.path.join(root_path, model_name)
        trained_model = train_and_save(
            train_X,
            train_y,
            trained_model,
            model_path
        )
        scores[model_path] = get_score_record(test_X, test_y, trained_model)
    scores = sorted(scores.items(), key=lambda x: x[1]["test_mse"], reverse=True)

    return scores

def get_top_features(estimator, features):
    """get top important feature names

    :param estimator: Estimator: trained model
    :param features: Array: all feature names
    :return: Array: top feature names
    """
    try:
        importance = features[np.argsort(-np.abs(estimator.coef_))][0, :TOP_FEATURES]
    except IndexError:
        importance = features[np.argsort(-np.abs(estimator.coef_))][:TOP_FEATURES]
    except AttributeError:
        importance = features[np.argsort(-np.abs(estimator.feature_importances_))][:TOP_FEATURES]
    return importance

def voting_feature(feature_importance_df):
    """ranking features by voting where first item
    gets weight 5 and last gets 1.

    :param feature_importance_df: DataFrame: (5,n) feature importance
    :return: list: tuple with feature and weight
    """
    importance_vote = {}
    for column in feature_importance_df.columns:
        importance_dict = dict(zip(feature_importance_df[column].tolist(), np.arange(TOP_FEATURES, 0, -1)))
        for key, value in importance_dict.items():
            if key in importance_vote:
                importance_vote[key] += value
            else:
                importance_vote[key] = value
    top_features = sorted(importance_vote.items(), key=lambda x: x[1], reverse=True)
    return top_features
