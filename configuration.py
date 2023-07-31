import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = 'data/'
STATS_PATH = 'stats/'
PLOTS_PATH = 'plots/'

features_file = f"{STATS_PATH}selected_features.txt"

"""Feature selection initial starting range for alpha parameter"""
fs_grid_params = {'alpha': np.arange(0.00001, 0.0001, 0.00001)}
FOLD_K = 5
NumberOfConfig = 50

Models_grid_params = [#"""XGBoost model"""
                        {'max_depth': [6, 9, 12],
                        'learning_rate': [0.005, 0.01, 0.1],
                        'subsample': [0.65, 0.7, 0.75, 0.8, 0.85],
                        'colsample_bytree': [0.65, 0.7, 0.75],
                        'min_child_weight': [0.5, 1.0, 3.0],
                        'gamma': [0, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0],
                        'n_estimators': [1000, 1500, 2000, 2500, 3000],
                        'eval_metric': ['auc'],
                        'tree_method': ['gpu_hist'],
                        'predictor': ['gpu_predictor'],
                        'objective': ['binary:logistic'],
                        'num_parallel_tree': [10],
                        'use_label_encoder': [False]},

                        #"""RandomForest model"""
                        {'n_jobs': [60],
                         'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13],
                         'max_features': ['sqrt'],
                         'min_samples_leaf': [1, 2, 3, 4, 5],
                         'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
                         'n_estimators': [1000, 1500, 2000, 2500, 3000, 3500]}
                    ]

Models = [XGBClassifier, RandomForestClassifier]
Models_Names = ["XGBoost", "RandomForest"]

utilize_models = [0, 1]
