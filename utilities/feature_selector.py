""""####################################################################################################################
Author: TBD
E-mail: TBD
-----------------------------------
Automatic feature selection model, which takes data and manage to automatically fine tune lasso over dynamic range of alpha
and returns list of selected features.
####################################################################################################################"""

import os, ast, sys, time, random
import numpy as np

from tqdm import tqdm
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from collections import defaultdict, Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import configuration as cnf
GLOBAL_LOSS = []
GLOBAL_ALPHA = []
GLOBAL_SLOPE = []
GLOBAL_UPDATE = []
#################################################################################################
##### TESTING SGD################################################################################
#################################################################################################
import matplotlib.pyplot as plt
def rmse(actual, pred):
    return np.sqrt(((pred - actual)**2).mean())
def cost_func(data, target, alpha, model):
    print('\t\tConst function')
    #model = Lasso(max_iter=500000, alpha=alpha)
    model.fit(data, target)
    #return rmse(rest_y, model.predict(rest_x))
    return rmse(target, model.predict(data))
def compute_slope(X, Y, A, H, model):
    global GLOBAL_LOSS
    #rmse1 = cost_func(X, Y, A + H, RX, RY)
    #rmse2 = cost_func(X, Y, A, RX, RY)
    original_error = cost_func(X, Y, A, model)
    step_model = Lasso(max_iter=500000, alpha=A + H)

    #print(f'\t\t\trmse1: {rmse1} rmse2: {rmse2}')
    GLOBAL_LOSS.append(original_error)
    return (cost_func(X, Y, A + H, step_model) - original_error) / H

#################################################################################################
######UNTIL HERE #############################################################################
#################################################################################################

"""Feature selection class"""
class FeatureSelector:
    def __init__(self, params=None, output_path=None,
                       stratified=True, shuffle=True,
                       verbose=True):

        self.verbose = verbose
        self.stratified = stratified
        self.shuffle = shuffle

        """Get list of searchable alpha params and sort them in reverse order"""
        self.params = params if params is not None else cnf.fs_grid_params['alpha']
        self.params.sort()
        self.params = self.params[::-1]

        self.param_history = []
        self.slope_history = []

        self.output_path = output_path
        self.learning_rate = 0.001
        self.h = 1e-10

    #################################################################################################
    #########################################TESTING#################################################
    def __sgd_feature_selection(self, input_X, input_Y, alpha, rest_X, rest_Y,number_of_steps=30):
        global GLOBAL_ALPHA
        global GLOBAL_LOSS
        global GLOBAL_SLOPE
        global GLOBAL_UPDATE

        best_alpha = alpha
        scaler = StandardScaler()


        input_X = scaler.fit_transform(input_X)
        #rest_X = scaler.transform(rest_X)

        alpha_path = []
        slope_path = []
        lr_path = []
        lr_update_cnt = 0
        model = Lasso(max_iter=500000, alpha=alpha, warm_start=True)

        #for step in range(number_of_steps):
        for step in range(number_of_steps):
            lr_path.append(self.learning_rate)
            alpha_path.append(alpha)

            #if len(alpha_path) > 3 and ( ((alpha_path[-1] - alpha_path[-2]) > 0) !=  ((alpha_path[-2] - alpha_path[-3]) > 0)):
            #    old = self.learning_rate
            #    self.learning_rate = 3*(self.learning_rate / 4) if (alpha_path[-1] - alpha_path[-2]) < 0 else 4*(self.learning_rate / 3)
            #    #self.learning_rate = self.learning_rate / 10 if (alpha_path[-1] - alpha_path[-2]) < 0 else self.learning_rate * 10
            #    print(f'\t\t Learning rate update {old} --> {self.learning_rate}')
            start_time = time.time()
            print(f'Shape {input_X.shape} and alpha: {alpha}')
            slope = compute_slope(input_X, input_Y, alpha, self.learning_rate, model)
            print(f'time spend: {(time.time() - start_time)/60} minutes')
            slope_path.append(slope)
            GLOBAL_ALPHA.append(alpha)
            GLOBAL_SLOPE.append(slope)
            print(f'Alpha: {alpha} slope: {slope} learning rate: {self.learning_rate}')

            if len(slope_path) > 2:
                last_1_slope_pos = slope_path[-1] > 0
                last_2_slope_pos = slope_path[-2] > 0
                if last_1_slope_pos != last_2_slope_pos:
                    #In this case we have angle change from positive to negative or from negative to positive. Such happens with large learning rate and step is too big
                    self.learning_rate = 2 * (self.learning_rate / 4)
                elif np.abs(slope > 100):
                    self.learning_rate = 8 *(self.learning_rate / 4)


            if np.abs(slope) <= 1e-10:#alpha < 0 or alpha < h or slope < 0.001:
                print('\t\t -- Converged !')
                GLOBAL_UPDATE.append(0)
                return True, alpha
            else:
                best_alpha = alpha
            #alpha_path.append(alpha)

            new_alpha = alpha - (self.learning_rate * np.sign(slope))
            #print(f'New ALPHA : {new_alpha} LR : {self.learning_rate}')
            if new_alpha < 0:
                #In case when the new alpha is negative we need to decrease the learning rate and re-compute
                #new_alpha until the alpha start to be positive number
                print(f'\t\tALPHA DIV')
                alpha = alpha / 2
            else:
                alpha = new_alpha



            """
            #alpha_path, self.learning_rate = update_learning_rate(self.learning_rate, slope, alpha_path)
            if len(alpha_path) > 3 and ( ((alpha_path[-1] - alpha_path[-2]) > 0) !=  ((alpha_path[-2] - alpha_path[-3]) > 0)) and lr_update_cnt :
                old = self.learning_rate
                self.learning_rate = 3*(self.learning_rate / 4) if (alpha_path[-1] - alpha_path[-2]) < 0 else 4*(self.learning_rate / 3)
                alpha = sum(alpha_path[-3:]) / 3
                #self.learning_rate = self.learning_rate / 10 if (alpha_path[-1] - alpha_path[-2]) < 0 else self.learning_rate * 10
                print(f'\t\t Learning rate update {old} --> {self.learning_rate}')
                GLOBAL_UPDATE.append(0.01)
            else:
                #alpha = np.abs(alpha - (self.learning_rate * slope))
                new_alpha = alpha - (self.learning_rate * slope)
                if new_alpha < 0:
                    self.learning_rate = 3*(self.learning_rate / 4)
                else:
                    alpha = new_alpha
                GLOBAL_UPDATE.append(0)
            """


        print(f'SGD best alpha: {best_alpha}')
        return False, best_alpha

    def __feature_selection_round_SGDtesting(self, X, Y):
        print('In SGD testing method')
        global GLOBAL_ALPHA
        global GLOBAL_LOSS
        global GLOBAL_SLOPE
        global GLOBAL_UPDATE

        """Check if feature selection is already done, we can simply load the results"""
        if os.path.exists(cnf.features_file):
            """Load already selected features """
            selected_features = ast.literal_eval(open(cnf.features_file, "r").read())
            return selected_features

        alpha = 0.01
        alpha_history = []
        convergence = False
        """Minimum number of batch repetitions, required in case of the binary classification with high imbalance. 
        In other case SGD make decision on the single pass until the  convergences achieved"""
        minimum_repetitions = 1 #Number of minimum batched repetitions

        """Repeat feature selection in case of regression is equal to 1 since 
        we dont resample-undersamples the original samples and repetition is not required"""
        # for repetition in range(1 if self.cont_values else 5):
        if not self.cont_values:
            class_counter = list(Counter(Y).values())

            minimum_repetitions = int(max(class_counter) / min(class_counter))
            if minimum_repetitions < 1:
                minimum_repetitions = 1
            print(f'Minimum repetitions set to: {minimum_repetitions}')

            undersample = RandomUnderSampler(sampling_strategy='not minority')

        start_time = time.time()
        converged_bool = False
        while True:

            if self.cont_values:
                """In case of continues values do not sample"""
                X_balanc, Y_balanc = X, Y
            else:
                """ In case of categorical values (binary or multiclass), balance samples between classes"""
                """resample all classes with the exception of minority class"""
                #undersample = RandomUnderSampler(sampling_strategy='not minority')
                X_balanc, Y_balanc = undersample.fit_resample(X, Y)
                rest_ind = [ind for ind in list(X.index) if ind not in set(X_balanc.index)]
                X_rest, Y_rest = X.iloc[rest_ind], Y.iloc[rest_ind]

            convergence, alpha = self.__sgd_feature_selection(X_balanc, Y_balanc, alpha, X_rest, Y_rest)
            alpha_history.append(alpha)


            if minimum_repetitions > 0:
                minimum_repetitions -= 1
            if convergence and not converged_bool:
                converged_bool = True
            ##################################!!!!!!!#################
            #if convergence or minimum_repetitions == 0:
            #    break
            ############################!!!!!!##
            if convergence and minimum_repetitions == 0:
                break
            if not converged_bool:
                self.learning_rate -= (self.learning_rate / 2)

        """find alpha that was selected as best most of the times"""
        # best_alpha = [(alpha, len(lasso_coef[alpha])) for alpha in lasso_coef]
        # best_alpha.sort(key=lambda t:t[1], reverse=True)
        # best_alpha = best_alpha[0][0]
        print(f'History: {alpha_history}')
        best_alpha=alpha

        if self.verbose: print(f'Best lasso alpha:{best_alpha} ')
        print(f'Time spend: {(time.time() - start_time) / 60}')
        x_time = list(range(len(GLOBAL_LOSS)))
        plt.plot(x_time, GLOBAL_LOSS, label='LOSS')
        plt.plot(x_time, GLOBAL_ALPHA, label='ALPHA')
        #plt.plot(x_time, GLOBAL_SLOPE, label='Slope')
        plt.plot(x_time, GLOBAL_UPDATE, label='Update')
        plt.legend()
        plt.show()

        self.best_alpha = best_alpha
        model_L = Lasso(max_iter=2000000, alpha=best_alpha)
        scaler = StandardScaler()

        """Fit the lasso model with best found alpha and the entire data used in feature selection"""
        model_L.fit(scaler.fit_transform(X), Y)

        if self.verbose:
            f_out = open(f'{self.output_path}stats/log_lasso_alpha.txt', "w+")
            f_out.write(f'Best alpha:{best_alpha}\n')
            f_out.close()

        selected_features = [feature for feature, coef in zip(X.columns.to_list(), model_L.coef_) if coef != 0]
        print(f'Selected {len(selected_features)} features.')

        sys.exit(-1)

        # self.feature_transl = dill.load(open("stats/feat_translate.dill", "rb"))
        if self.verbose:
            f_out = open(f'{self.output_path}stats/selected_features.txt', "w+")
            f_out.write(f'{selected_features}')
            f_out.close()

        return selected_features

    def __feature_selection_round_STEPtesting(self, X, Y):
        print('In My STEP testing method')
        global GLOBAL_ALPHA
        global GLOBAL_LOSS
        global GLOBAL_SLOPE
        global GLOBAL_UPDATE

        """Check if feature selection is already done, we can simply load the results"""
        if os.path.exists(cnf.features_file):
            """Load already selected features """
            selected_features = ast.literal_eval(open(cnf.features_file, "r").read())
            return selected_features

        alpha = 0.01
        alpha_history = []
        convergence = False
        """Minimum number of batch repetitions, required in case of the binary classification with high imbalance. 
        In other case SGD make decision on the single pass until the  convergences achieved"""
        minimum_repetitions = 1 #Number of minimum batched repetitions

        """Repeat feature selection in case of regression is equal to 1 since 
        we dont resample-undersamples the original samples and repetition is not required"""
        # for repetition in range(1 if self.cont_values else 5):
        if not self.cont_values:
            class_counter = list(Counter(Y).values())

            minimum_repetitions = int(max(class_counter) / min(class_counter))
            if minimum_repetitions < 1:
                minimum_repetitions = 1
            if minimum_repetitions > 10:
                minimum_repetitions = 10
            print(f'Minimum repetitions set to: {minimum_repetitions}')

            undersample = RandomUnderSampler(sampling_strategy='not minority')

        start_time = time.time()
        converged_bool = False

        self.params.sort()

        """Create repetition resampling indexes in order test model over the identical selected samples"""
        rep_indexes = []

        if self.cont_values:
            """In case of continues values do not sample"""
            rep_indexes.append(list(X.index))
        else:
            """ In case of categorical values (binary or multiclass), balance samples between classes"""
            """resample all classes with the exception of minority class"""
            target_count = Counter(Y)
            target_info = [(item, target_count[item]) for item in target_count]
            target_info.sort(key=lambda t:t[1], reverse=False)

            _ , sample_per_class = target_info[0]
            ignore_targets = [target for target, samples in target_info if samples < (sample_per_class * minimum_repetitions)]

            class_indexes = {target : list(Y[Y==target].index) for target,_ in target_info}

            for rep in range(minimum_repetitions):
                selected_indexes = []
                for target,_ in target_info:
                    print(f'target: {target} len: {len(class_indexes[target])} sample per class: {sample_per_class}')
                    class_sample = random.sample(class_indexes[target], sample_per_class)
                    selected_indexes += class_sample

                    if target not in ignore_targets:
                        """In case of not the mninority class, we need to remove selected samples"""
                        class_indexes[target] = list(set(class_indexes[target]) - set(class_sample))
                rep_indexes.append(selected_indexes)


        print('Start alpha testing')
        round_alpha_loss = []
        """Alpha parameters in searching range are sorted from minimum to maximum"""
        for alpha in self.params:
            model = Lasso(max_iter=500000, alpha=alpha, warm_start=True)
            scaler = StandardScaler()

            tmp_loss = 0.0
            for rep in tqdm(range(len(rep_indexes))):
                start_time = time.time()
                print(f'Iter: {rep} of {len(rep_indexes)} alpha: {alpha}')
                X_balanc, Y_balanc = X.iloc[rep_indexes[rep]], Y.iloc[rep_indexes[rep]]

                print(f'\t\tshape: {X_balanc.shape}')
                scaled_X = scaler.fit_transform(X_balanc)
                error = cost_func(scaled_X, Y_balanc, alpha, model)
                print(f'\t\terror: {error}')
                print(f'Time spend: {(time.time() - start_time) / 60} minutes')
                tmp_loss += error

            tmp_loss /= len(rep_indexes)
            round_alpha_loss.append((tmp_loss, alpha))
            print(f'\t\tTested alpha: {alpha} with avg loss: {tmp_loss}')

            if len(round_alpha_loss) > 0 and  round_alpha_loss[-1][0] > round_alpha_loss[-2][0]:
                """In case of loss increase stop"""
                print('Stop criteria')
                break
        round_alpha_loss.sort(key=lambda t:t[0], reverse=False)
        best_alpha = round_alpha_loss[0][1]

        new_range = self.create_new_grid(best_alpha)
        if new_range is not None:
            if self.verbose: print(f"Lasso found boundary alpha: {best_alpha} from {self.params}")

            """Update the alpha range"""
            self.params = new_range
            if self.verbose: print(f"New alpha range: {self.params}")
            return None

        if self.verbose: print(
            f'Best lasso alpha:{best_alpha} from {cnf.fs_grid_params["alpha"]} and iterational range: {self.params}')

        self.best_alpha = best_alpha
        model_L = Lasso(max_iter=2000000, alpha=best_alpha)
        scaler = StandardScaler()

        """Fit the lasso model with best found alpha and the entire data used in feature selection"""
        model_L.fit(scaler.fit_transform(X), Y)

        if self.verbose:
            f_out = open(f'{self.output_path}stats/log_lasso_alpha.txt', "w+")
            f_out.write(f'Best alpha:{best_alpha}\n')
            f_out.close()

        selected_features = [feature for feature, coef in zip(X.columns.to_list(), model_L.coef_) if coef != 0]

        # self.feature_transl = dill.load(open("stats/feat_translate.dill", "rb"))
        if self.verbose:
            f_out = open(f'{self.output_path}stats/selected_features.txt', "w+")
            f_out.write(f'{selected_features}')
            f_out.close()

        return selected_features

    #################################################################################################
    #################################################################################################

    def __feature_selection_round(self, X, Y, executions=0):
        print('In my kfold method')
        """Check if feature selection is already done, we can simply load the results"""
        if os.path.exists(cnf.features_file):
            """Load already selected features """
            selected_features = ast.literal_eval(open(cnf.features_file, "r").read())
            # self.X_holdOUT = self.X_holdOUT[self.selected_features]
            # self.X_visible = self.X_visible[self.selected_features]
            return selected_features

        lasso_coef = defaultdict(lambda: 0)

        """sort alpha from higher values to lowest"""
        self.params.sort()
        self.params = self.params[::-1]

        errors = defaultdict(lambda: [])

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

            kfolds = StratifiedKFold(n_splits=2)
            indexes = [(train_index, test_index) for train_index, test_index in kfolds.split(X_balanc, Y_balanc)]

            scaler = StandardScaler()
            # X_balanc = scaler.fit_transform(X_balanc)
            for ind, alpha in enumerate(self.params):
                model = Lasso(max_iter=500000, alpha=alpha)
                error = 0.0

                for i in range(len(indexes)):
                    print(f"Fold {i}:")
                    X_train, Y_train = scaler.fit_transform(X_balanc.iloc[indexes[i][0]]), Y_balanc.iloc[indexes[i][0]]
                    X_val, Y_val = scaler.transform(X_balanc.iloc[indexes[i][1]]), Y_balanc.iloc[indexes[i][1]]

                    print(f'\tAlpha: {alpha}')
                    model.fit(X_train, Y_train)
                    error += rmse(Y_val, model.predict(X_val))
                    # errors[alpha].append(rmse(Y_val, model.predict(Y_val)))

                error /= len(indexes)

                # error = cost_func(X_balanc, Y_balanc, alpha, model)
                errors[alpha].append(error)
                print(f'Rep: {repetition} Alpha : {alpha} with error : {error}')

        """find alpha that was selected as best most of the times"""
        path = [(alpha, sum(errors[alpha]) / len(errors[alpha]), ind) for ind, alpha in enumerate(self.params)]
        path.sort(key=lambda t: t[1])
        best_alpha = path[0][0]
        best_alpha_ind = path[0][2]

        new_range = self.create_new_grid_by_index(best_alpha_ind)  # self.create_new_grid(best_alpha)
        if new_range is not None:
            if self.verbose: print(f"Lasso found boundary alpha: {best_alpha} from {self.params}")

            """Update the alpha range"""
            self.params = new_range
            if self.verbose: print(f"New alpha range: {self.params}")
            return None

        if self.verbose: print(
            f'Best lasso alpha:{best_alpha} from {cnf.fs_grid_params["alpha"]} and iterational range: {self.params}')

        self.best_alpha = best_alpha
        model_L = Lasso(max_iter=2000000, alpha=best_alpha)
        scaler = StandardScaler()

        """Fit the lasso model with best found alpha and the entire data used in feature selection"""
        model_L.fit(scaler.fit_transform(X), Y)

        if self.verbose:
            f_out = open(f'{self.output_path}stats/log_lasso_alpha.txt', "w+")
            f_out.write(f'Best alpha:{best_alpha}\n')
            f_out.close()

        selected_features = [feature for feature, coef in zip(X.columns.to_list(), model_L.coef_) if coef != 0]
        print(f'Selected {len(selected_features)} features')

        # self.feature_transl = dill.load(open("stats/feat_translate.dill", "rb"))
        if self.verbose:
            f_out = open(f'{self.output_path}stats/selected_features.txt', "w+")
            f_out.write(f'{selected_features}')
            f_out.close()

        return selected_features

    def __feature_selection_round_old(self, X, Y, executions=0):
        print('In Original step method')
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
                ('scaler', MinMaxScaler()),#StandardScaler()),
                ('model', Lasso(max_iter=500000))
            ])
            kfolds = StratifiedKFold(cnf.FOLD_K)
            search = GridSearchCV(pipeline,
                                  {'model__alpha': self.params},
                                  cv=kfolds.split(X_balanc, Y_balanc),
                                  scoring="neg_mean_squared_error",
                                  verbose=3,
                                  n_jobs=120
                                  )
            search.fit(X, Y)
            # lasso_coef[search.best_params_['model__alpha']].append(search.best_estimator_.named_steps['model'].coef_)
            lasso_coef[search.best_params_['model__alpha']] += 1


        """find alpha that was selected as best most of the times"""
        # best_alpha = [(alpha, len(lasso_coef[alpha])) for alpha in lasso_coef]
        # best_alpha.sort(key=lambda t:t[1], reverse=True)
        # best_alpha = best_alpha[0][0]
        best_alpha = max(lasso_coef, key=lasso_coef.get)

        new_range = self.create_new_grid(best_alpha)
        if new_range is not None:
            if self.verbose: print(f"Lasso found boundary alpha: {best_alpha} from {cnf.fs_grid_params['alpha']}")

            """Update the alpha range"""
            self.params = new_range
            if self.verbose: print(f"New alpha range: {cnf.fs_grid_params['alpha']}")
            return None

        if self.verbose: print(f'Best lasso alpha:{best_alpha} from {cnf.fs_grid_params["alpha"]} and iterational range: {self.params}')

        self.best_alpha = best_alpha
        model_L = Lasso(max_iter=2000000, alpha=best_alpha)
        scaler = StandardScaler()

        """Fit the lasso model with best found alpha and the entire data used in feature selection"""
        model_L.fit(scaler.fit_transform(X), Y)

        if self.verbose:
            f_out = open(f'{self.output_path}stats/log_lasso_alpha.txt', "w+")
            f_out.write(f'Best alpha:{best_alpha}\n')
            f_out.close()

        selected_features = [feature for feature, coef in zip(X.columns.to_list(), model_L.coef_) if coef != 0]


        # self.feature_transl = dill.load(open("stats/feat_translate.dill", "rb"))
        if self.verbose:
            f_out = open(f'{self.output_path}stats/selected_features.txt', "w+")
            f_out.write(f'{selected_features}')
            f_out.close()

        return selected_features



    def get_features(self, X, Y):
        if self.verbose: print(f'{datetime.now()} Start fine-tuning of feature selection')
        
        """Local variables that used in order to identify if the dataset is for binary classification or for regression"""
        self.binary_class = True if len(set(Y)) == 2 else False
        self.cont_values = True if type(list(Y)[0]) == np.float64 or len(set(Y)) > 15 else False

        if os.path.isfile(f'{self.output_path}stats/selected_features.txt'):
            if self.verbose: print("Feature selector skip fs since found previously selected features.")
            features_found = ast.literal_eval(open(f'{self.output_path}stats/selected_features.txt', "r").read())
        else:
            features_found = None
        executions = 0
        initial_features = X.shape[1]
        while features_found is None:
            """Executions value now not used, bu in future can be used to limit number of executions"""
            """SGD Testing """
            #features_found = self.__feature_selection_round_SGDtesting(X, Y)

            """TESTING Original steps but implemented by Alex"""
            #features_found = self.__feature_selection_round_STEPtesting(X, Y)

            """New implementation of range testing"""
            features_found = self.__feature_selection_round(X, Y)

            """DEFAULT OLD """
            #features_found = self.__feature_selection_round_old(X, Y, executions=executions)
        #sys.exit(-1)

        if self.verbose:
            print(f'{datetime.now()} Done\n' +
                  f'\tSelected {len(features_found )} of {initial_features} features.')

        return features_found


    """Dynamic creation of the parameter range based on the previous range and selected best alpha
    In case of lower value create new range over smallest values with extra padding. Similarly works on high ranges"""
    def create_new_grid(self, found):
        if len(self.params) < 2:
            return None

        self.params.sort()
        step = self.params[1] - self.params[0]

        """If found parameter for lasso belong to the lower or upper bound we need to extend the range and check again"""
        if found in [self.params[0], self.params[-1]]:
            print('In range')
            """Minimum case """
            if found == self.params[0]:
                end = self.params[1]
                start = self.params[0] - self.params[1]

                """If the lower bound become negative, we reduce the step and compute the range again with small overlap over the high values"""
                if end - (len(self.params) * step) < 0.0:

                    start = end - (len(self.params) * step)
                    end += step

            elif found == max(self.params):
                start = self.params[-2]  # found
                end = start + (len(self.params) * step)
            print(f'start: {start} end: {end} step: {step}')
            """Check if start or end is too high or too low"""
            if start <= 0.000001 or end > 10 or step <= 0.000001:
                return None
            """Return new updated range"""
            step = (end - start) / len(self.params)
            return np.arange(start, end, step)
        else:
            print('No in range')
            """In other case when found value belong between upper and lower bound we have found best local value"""
            return None

    """Dynamic creation of the parameter range based on the previous range and selected best alpha
        In case of lower value create new range over smallest values with extra padding. Similarly works on high ranges"""
    def create_new_grid_by_index(self, ind):
        self.params.sort()
        self.params = self.params[::-1]
        step = round(np.abs(self.params[1] - self.params[0]), 7)
        print(f'Ind: {ind} step : {step}')
        """If found parameter for lasso belong to the lower or upper bound we need to extend the range and check again"""
        if (ind > 0 and ind < len(self.params) - 1):
            print('Found middle solution')
            """In other case when found value belong between upper and lower bound we have found best local value"""
            return None
        elif ind < 0 or ind > len(self.params) - 1:
            raise Exception(
                f'New grid creation, index value out of possible range index: {ind} max range: {len(self.params) - 1}')
        print(f'In range ind: {ind} alpha : {self.params[ind]}')
        slope = 1 if ind == len(self.params) - 1 else -1
        print(f'Slope : {slope}')

        self.param_history = []
        self.slope_history = []
        if len(self.slope_history) > 1 and self.slope_history[-1] != slope:
            """In case when model take back and forth steps wee need to focus in between those steps"""
            print('Back and forth')
            boundaries = [self.param_history[-1], self.params[ind]]
            start = round(min(boundaries) - step/2, 7)
            end = round(max(boundaries) + step/2, 7)
            if start < 0:
                start = 0.0
            step = round((end - start) / (len(self.params)*2), 7)
            print(f'start: {start} end: {end} step: {step}')
            self.slope_history = []
            self.param_history = []

        else:
            start = round(self.params[ind] + (slope * (step / 2)), 7)
            end = round(start - slope * (len(self.params) * step), 7)
            # while end < 0:
            #    print('Loop')
            #    end += round(step/2, 7)
            if end < 0:
                end = 0.0
            step = round((end - start) / len(self.params), 7)
            print(f'start: {start} end: {end} step: {step}')
            """Check if start or end is too high or too low"""
            if (start <= 1e-6 or end > 10 or np.abs(step) <= 1e-6 or
                    (slope > 0 and end >= start) or (slope < 0 and end <= start)):
                print('Here')
                return None
            self.slope_history.append(slope)
            self.param_history.append(self.params[ind])
        """Return new updated range"""
        new_range = np.arange(start, end, step)
        new_range.sort()
        return new_range[::-1]