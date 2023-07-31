""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Automatic feature selection model, which takes data and manage to automatically fine tune lasso over dynamic range of alpha
and returns list of selected features.
####################################################################################################################"""

import os, ast
import numpy as np

from datetime import datetime
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import configuration as cnf


"""Dynamic creation of the parameter range based on the previous range and selected best alpha
In case of lower value create new range over smallest values with extra padding. Similarly works on high ranges"""
def create_new_grid(values, found):
    if len(values) < 2:
        return None

    values.sort()
    step = values[1] - values[0]

    """If found parameter for lasso belong to the lower or upper bound we need to extend the range and check again"""
    if found in [values[0], values[-1]]:
        """Minimum case """
        if found == values[0]:
            end = values[1]  # found
            start = values[0] - values[1]  # end - (len(values) * step)

            """If the lower bound become negative, we reduce the step and compute the range again with small overlap over the high values"""
            if end - (len(values) * step) < 0.0:
                # step /= 10
                start = end - (len(values) * step)
                end += step  # step*(int(0.5* len(values)) + 1 )

        elif found == max(values):
            start = values[-2]  # found
            end = start + (len(values) * step)
        """Check if start or end is too high or too low"""
        if start <= 0.000001 or end > 10 or step <= 0.000001:
            return None
        """Return new updated range"""
        step = (end - start) / len(values)
        return np.arange(start, end, step)
    else:
        """In other case when found value belong between upper and lower bound we have found best local value"""
        return None

"""Feature selection class"""
class FeatureSelector:
    def __init__(self, stratified=True, shuffle=True, verbose=True):

        self.verbose = verbose
        self.stratified = stratified
        self.shuffle = shuffle

    def __feature_selection_round(self, X, Y, executions=0):
        """Check if feature selection is already done, we can simply load the results"""
        if os.path.exists(cnf.features_file):
            """Load already selected features """
            selected_features = ast.literal_eval(open(cnf.features_file, "r").read())

            #self.X_holdOUT = self.X_holdOUT[self.selected_features]
            #self.X_visible = self.X_visible[self.selected_features]
            return selected_features

        lasso_coef = defaultdict(lambda: 0)

        """Repeat feature selection in case of regression is equal to 1 since 
        we dont resample-undersamples the original samples and repetition is not required"""
        for repetition in range(1 if self.cont_values else 5):
            if self.cont_values:
                """In case of continues values do not sample"""
                X_balanc, Y_balanc = X, Y
            else:
                """ In case of categorical values (binary or multiclass), balance samples between classes"""
                """resample all classes with the exception of minority class"""
                undersample = RandomUnderSampler(sampling_strategy='not minority')
                X_balanc, Y_balanc = undersample.fit_resample(X, Y)

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(max_iter=1000000))
            ])
            kfolds = StratifiedKFold(cnf.FOLD_K)
            search = GridSearchCV(pipeline,
                                  {'model__alpha': cnf.fs_grid_params['alpha']},
                                  cv=kfolds.split(X_balanc, Y_balanc),
                                  scoring="neg_mean_squared_error",
                                  verbose=3,
                                  n_jobs=60
                                  )
            search.fit(X, Y)
            # lasso_coef[search.best_params_['model__alpha']].append(search.best_estimator_.named_steps['model'].coef_)
            lasso_coef[search.best_params_['model__alpha']] += 1

        """find alpha that was selected as best most of the times"""
        # best_alpha = [(alpha, len(lasso_coef[alpha])) for alpha in lasso_coef]
        # best_alpha.sort(key=lambda t:t[1], reverse=True)
        # best_alpha = best_alpha[0][0]
        best_alpha = max(lasso_coef, key=lasso_coef.get)

        new_range = create_new_grid(cnf.fs_grid_params['alpha'], best_alpha)
        if new_range is not None:
            if self.verbose: print(f"Lasso found boundary alpha: {best_alpha} from {cnf.fs_grid_params['alpha']}")

            """Update the alpha range"""
            cnf.fs_grid_params['alpha'] = new_range
            if self.verbose: print(f"New alpha range: {cnf.fs_grid_params['alpha']}")
            return None

        if self.verbose: print(f'Best lasso alpha:{best_alpha} from {cnf.fs_grid_params["alpha"]}')

        model_L = Lasso(max_iter=1000000, alpha=best_alpha)
        scaler = StandardScaler()

        """Fit the lasso model with best found alpha and the entire data used in feature selection"""
        model_L.fit(scaler.fit_transform(X), Y)


        f_out = open(f'{cnf.STATS_PATH}log_lasso_alpha.txt', "w+")
        f_out.write(f'Best alpha:{best_alpha}\n')
        f_out.close()

        selected_features = [feature for feature, coef in zip(X.columns.to_list(), model_L.coef_) if coef != 0]


        # self.feature_transl = dill.load(open("stats/feat_translate.dill", "rb"))
        f_out = open(cnf.features_file, "w+")
        f_out.write(f'{selected_features}')
        f_out.close()

        return selected_features



    def get_features(self, X, Y):
        if self.verbose: print(f'{datetime.now()} Start fine-tuning of feature selection')

        """Local variables that used in order to identify if the dataset is for binary classification or for regression"""
        self.binary_class = True if len(set(Y)) == 2 else False
        self.cont_values = True if type(list(Y)[0]) == np.float64 or len(set(Y)) > 15 else False

        features_found = None
        executions = 0
        initial_features = X.shape[1]
        while features_found is None:
            """Executions value now not used, bu in future can be used to limit number of executions"""
            features_found = self.__feature_selection_round(X, Y, executions=executions)

        if self.verbose:
            print(f'{datetime.now()} Done\n' +
                  f'\tSelected {len(features_found )} of {initial_features} features.')

        return features_found
