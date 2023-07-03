""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Parameter fine-tuning and Feature selection for ML model.
####################################################################################################################"""

import os, ast, dill
import numpy as np
import pandas as pd


from plotting import plot_roc_curves, plot_shap_figure, plot_confusion_figure
from datetime import datetime
"""Models that we use"""
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from utilities.DataLoading import DataLoading
from utilities.Model import Model

from imblearn.under_sampling import RandomUnderSampler


from collections import defaultdict

DATA_PATH = 'data/'
STATS_PATH = 'stats/'
PLOTS_PATH = 'plots/'



"""Custom evaluation function that compute ROC-AUC score. 
   It capable to compute proper ROC-AUC score for binary and multy-class classifications
   Also, in case when model not able to provide probabilities, 
   it able to compute the accurate score for binary and multy-class classification
"""
def roc_auc_evaluation(y_true, y_pred_prob):
    """In case of binary classification when function receive single vector"""

    """in case of binary classification even if we recieve the predicted Y or the probabilities of Y"""
    if len(set(y_true)) <= 2 or (len(y_pred_prob.shape) == 2 and y_pred_prob.shape[1] == 2): #Binary classification problem
        roc_auc_score(y_true, y_pred_prob if len(y_pred_prob.shape) == 1 else y_pred_prob[:,1])

    else:#Non binary classification
        """In case when we have multi-class with Y-pred is actually probabilities for each class"""
        if len(y_pred_prob.shape) == 2 and y_pred_prob.shape[1] >= 2:
            return roc_auc_score(y_true, y_pred_prob, average="weighted", multi_class='ovr')
        elif len(y_pred_prob.shape) == 1:
            """In case of multi-class with the predicted Y (non-probs) we need to compute ROC-AUC by hand"""
            avg_score = []
            for outcome_value in set(y_true):
                original = [1 if item == outcome_value else 0 for item in list(y_true)]
                predicted = [1 if item == outcome_value else 0 for item in list(y_pred_prob)]
                avg_score.append(roc_auc_score(original, predicted))
            return sum(avg_score) / len(avg_score)

    #if y_pred_prob.shape[1] == 2:
    #
    #    """If we get float, then the predicted outcome is probabilities"""
    #    if type(y_pred_prob[0]) == np.float64:
    #        return roc_auc_score(y_true, y_pred_prob[:,1])
    #
    #    """If the outcome is something else we need to compute multy-class roc-auc by hand"""
    #    avg_score = []
    #    for outcome_value in set(y_true):
    #        original = [1 if item == outcome_value else 0 for item in list(y_true)]
    #        predicted = [1 if item == outcome_value else 0 for item in list(y_pred_prob)]
    #        avg_score.append(roc_auc_score(original, predicted))
    #    return sum(avg_score) / len(avg_score)

    #"""In case of multy-class"""
    #if y_pred_prob.shape[1] > 2:
    #    return roc_auc_score(y_true, y_pred_prob,
    #                  average="weighted",
    #                  multi_class='ovr')
    #else:
    #    """In case of binary classification with both vectors"""
    #    return roc_auc_score(y_true, y_pred_prob[:, 1])
    raise Exception("ROC-AUC evaluation: no case found to process the data")


def create_new_grid(values, found):

    if len(values) < 2:
        return None

    values.sort()
    step = values[1] - values[0]

    """If found parameter for lasso belong to the lower or upper bound we need to extend the range and check again"""
    if found in [values[0], values[-1]]:
        if found == values[0]:
            end = found
            start = end - (len(values) * step)

            """If the lower bound become negative, we reduce the step and compute the range again with small overlap over the high values"""
            if end - (len(values) * step) < 0.0:
                step /= 10
                start = end - (len(values) * step)
                end += step*(int(0.5* len(values)) + 1 )

        elif found == max(values):
            start = found
            end = start + (len(values) * step)
        """Return new updated range"""
        return np.arange(start, end , step)
    else:
        """In other case when found value belong between upper and lower bound we have found best local value"""
        return None




class Piepeline:
    def __init__(self, NumberOfConfig, FS, Models, Models_grid_params,
                       fs_grid_params, K=5, stratified=True, scorer=None,
                       shuffle=True, verbose=True):
         
        self.verbose = verbose

        self.scorer = scorer

        """Loading the Data splited in train/test and hold-out portions"""
        self.X_visible, self.Y_visible, self.X_holdOUT, self.Y_holdOUT = DataLoading(data_path=DATA_PATH,
                                                                         verbose=self.verbose).load_dataset()

        """In case of binary classification:
        Check if dataset is imbalance, in case of imbalance we should fix class weights in models"""
        if len(set(self.Y_visible)) == 2 and sum(self.Y_visible == 0) != sum(self.Y_visible == 1):
            """Since our dataset is imbalanced in terms of targets we need to compute proper weights per outcome for each model"""

            neg_count = sum(self.Y_visible == 0)
            pos_count = sum(self.Y_visible == 1)

            for model_index in range(len(Models)):
                if Models[model_index] == XGBClassifier:
                    Models_grid_params[model_index]['scale_pos_weight'] = [neg_count / pos_count]
                elif Models[model_index] == RandomForestClassifier:
                    Models_grid_params[model_index]['class_weight'] = [{
                                                            0: (self.Y_visible.shape[0] / (2 * neg_count)),
                                                            1: (self.Y_visible.shape[0] / (2*pos_count))
                                                            }]
        elif len(set(self.Y_visible)) != 2:
            Models_grid_params[0]['objective'] = ['multi: softprob']
            Models_grid_params[0]['num_class'] = [len(set(self.Y_visible))]


        self.FS = FS
        
        self.models = [Model(nmbr_to_select=NumberOfConfig, configs_ranges=Models_grid_params[i], model=Models[i]) for i in range(len(Models))]
        self.features_file = f'{STATS_PATH}selected_features.txt'


        self.fs_grid_params = fs_grid_params
        self.K = K
        self.stratified = stratified
        self.shuffle = shuffle
        self.binary_class = True if len(set(self.Y_visible)) == 2 else False
        
        self.main()


    def feature_selection(self):

        if os.path.exists(self.features_file):
            self.selected_features = ast.literal_eval(open(self.features_file, "r").read())

            self.X_holdOUT = self.X_holdOUT[self.selected_features]
            self.X_visible = self.X_visible[self.selected_features]
            return

        lasso_coef = defaultdict(lambda: [])
        #binary = True if len(set(self.Y_visible)) == 2 else False
        cont_values = True if type(list(self.Y_visible)[0]) == np.float64 else False
        for repetition in range(1 if cont_values else 5):
            if not cont_values:
                """ In case of categorical values, balance samples between classes"""
                """resample all classes with the exception of minority class"""
                undersample = RandomUnderSampler(sampling_strategy='not minority')

                X, Y = undersample.fit_resample(self.X_visible, self.Y_visible)
            else:
                """In case of continues values do not sample"""
                X, Y = self.X_visible, self.Y_visible

            pipeline = Pipeline([
                               ('scaler', StandardScaler()),
                               ('model', Lasso(max_iter=1000000))
                           ])
            kfolds = StratifiedKFold(self.K)
            search = GridSearchCV(pipeline,
                                  {'model__alpha': self.fs_grid_params['alpha']},
                                  cv=kfolds.split(X, Y),
                                  #"""For binary classification use roc_auc scorer in other case of regression or multiclass use rmse"""
                                  scoring="roc_auc" if self.binary_class else "neg_mean_squared_error",
                                  verbose=3, n_jobs = 60
                                  )
            search.fit(X, Y)

            lasso_coef[search.best_params_['model__alpha']].append(search.best_estimator_.named_steps['model'].coef_)

        """find alpha that was selected as best most of the times"""
        best_alpha = [(alpha, len(lasso_coef[alpha])) for alpha in lasso_coef]
        best_alpha.sort(key=lambda t:t[1], reverse=True)

        best_alpha = best_alpha[0][0]
        new_range = create_new_grid(self.fs_grid_params['alpha'], best_alpha)
        if new_range is not None:
            print(f"Lasso found boundary alpha: {best_alpha} from {self.fs_grid_params['alpha']}")
            """Update the alpha range"""
            self.fs_grid_params['alpha'] = new_range
            print(f"New alpha range: {self.fs_grid_params['alpha']}")
            return self.feature_selection()

        print(f'Best lasso alpha:{best_alpha} from {self.fs_grid_params["alpha"]}')
        best_coef = sum(lasso_coef[best_alpha]) / len(lasso_coef[best_alpha])


        f_out = open(f'{STATS_PATH}log_lasso_alpha.txt', "w+")
        f_out.write(f'Best alpha:{best_alpha}\n')
        f_out.close()

        feature_names = self.X_visible.columns.to_list()
        self.selected_features = [feature_names[i] for i in range(len(best_coef)) if best_coef[i] != 0]
        
        #self.feature_transl = dill.load(open("stats/feat_translate.dill", "rb"))
        f_out = open(self.features_file, "w+")
        f_out.write(f'{self.selected_features}')
        f_out.close()

        self.X_holdOUT = self.X_holdOUT[self.selected_features]
        self.X_visible = self.X_visible[self.selected_features]
        


    """Return best configuration and average performance of best performance during K-Fold Cross Validation"""
    def fine_tune_models(self):

        self.data_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))
        performances = defaultdict(lambda: defaultdict(lambda: {"roc-auc-val": [], "roc-auc-train": []}))
            
        fold_ind = 0
        skfold = StratifiedKFold(n_splits=self.K, shuffle=self.shuffle)
        for train_index, val_index in skfold.split(self.X_visible, self.Y_visible):

            train_X, val_X = self.X_visible.iloc[train_index, :], self.X_visible.iloc[val_index, :]
            train_Y, val_Y = self.Y_visible.iloc[train_index, ], self.Y_visible.iloc[val_index, ]
           

            print(f'Fold index:{fold_ind+1} of {self.K}')

            for i, (model) in enumerate(self.models):
                print(f'Model: {i+1}')
                for model_config in tqdm(model.parameters):

                    model.create_model(model_config)

                    """Train model in train data, predict train data and predict validation data"""
                    YP_train, YP_val = model.train_predict(train_X, train_Y, val_X)

                    performances[i][f'{model_config}']["roc-auc-val"].append(roc_auc_evaluation(val_Y, YP_val))
                    performances[i][f'{model_config}']["roc-auc-train"].append(roc_auc_evaluation(train_Y, YP_train))

                    """Keep false positive rate, true positive rate for later plotting with shadowing for ROC-AUC curves"""
                    #fpr, tpr, _ = roc_curve(val_Y, YP_val)
                    #precision, recall, _ = precision_recall_curve(val_Y, YP_val)

                    #self.data_stats[i][fold_ind][f'{model_config}']["FP"] = fpr
                    #self.data_stats[i][fold_ind][f'{model_config}']["TP"] = tpr

            fold_ind += 1
        
        return performances
        

    """Return selected model based on best average performances during K-Fold Cross Validation"""
    def measure_and_select(self, performances=None):
        
        ready_models = []
        model_labels = ["XGBoost", "RandomForest"]
        model_indexes = [i+1 for i in range(len(model_labels))]
        

        train_roc_auc, val_roc_auc, holdout_roc_auc, decision_th, gmeans, best_configs = [],[],[],[],[],[]


        for model_index in range(len(self.models)):
            for config in performances[model_index]:
                for metric in performances[model_index][config]:
                    performances[model_index][config][metric] = sum(performances[model_index][config][metric]) / len(
                        performances[model_index][config][metric])

            """Find model configuration which provide maximum average ROC-AUC score over validation set"""
            model_best_config = [(config, performances[model_index][config]["roc-auc-val"]) for config in performances[model_index]]
            model_best_config.sort(key=lambda t: t[1], reverse=True)

            """Store this configuration for particular model"""
            best_configs.append(model_best_config[0][0])
            

            """Create model"""
            model = self.models[model_index]
            model.create_model(ast.literal_eval(best_configs[-1]))
            model.fit(self.X_visible, self.Y_visible)
            ready_models.append(model)
            
            hold_out_roc_auc = roc_auc_evaluation(self.Y_holdOUT,
                                             model.predict(self.X_holdOUT))

            """Binary classification requires adjust of the decision threashold"""
            if self.binary_class:
                """Compute and store figure with ROC-AUC performance"""
                fpr, tpr, th = roc_curve(self.Y_holdOUT,
                                      model.predict_proba(self.X_holdOUT)[:, 1])

                #hold_out_roc_auc = roc_auc_evaluation(self.Y_holdOUT,
                #                             model.predict(self.X_holdOUT))

                """calculate the g-mean for each threshold"""
                g_mean = np.sqrt(tpr * (1 - fpr))

                """locate the index of the larger g-mean"""
                ix = np.argmax(g_mean)

                decision_th.append(th[ix])
                gmeans.append(g_mean[ix])
            else:
                decision_th.append(-1.0)
                gmeans.append(-1.0)
            
            train_roc_auc.append(performances[model_index][best_configs[-1]]["roc-auc-train"] )
            val_roc_auc.append(performances[model_index][best_configs[-1]]["roc-auc-val"])
            holdout_roc_auc.append(hold_out_roc_auc)

        stats_df = pd.DataFrame([model_labels, model_indexes,
                                 train_roc_auc, val_roc_auc, holdout_roc_auc, 
                                 decision_th, gmeans, best_configs, 
                                 [f'{self.selected_features}' for i in model_indexes]]).T

        stats_df.columns = ['name', 'model_index', 'train_rocauc',
                            'valid_rocauc', 'holdout_rocauc', 
                            'decision_threshold', 'gmean', 'params',
                            'features']

        stats_df.to_csv(f'{STATS_PATH}pipeline_result.csv', index=False, sep='\t')


        """Identify model that performs better in VALIDATION dataset in average. This model will be selected as final model.
           WE DON"T UTILISE HOLD-OUT DATASET PERFORMANCE FOR FINAL MODEL SELECTION"""
        for model_ind in range(len(self.models)):
            for fold in self.data_stats[model_ind]:
                conf = list(self.data_stats[model_ind][fold].keys())[0]
                self.data_stats[model_ind][fold] = self.data_stats[model_ind][fold][conf]
        
        """Find best model index with highest avg roc-auc Validation score"""
        best_model_index = [(ind, performances[ind][best_configs[ind]]["roc-auc-val"]) for ind in range(len(performances))]
        best_model_index.sort(key=lambda t:t[1], reverse=True)
        best_model = ready_models[best_model_index[0][0]]

        
        if self.binary_class:
            print("TN\tFP\tFN\tTP")
            tn, fp, fn, tp = confusion_matrix(self.Y_holdOUT,
                [1 if pred > decision_th[0] else 0  for pred in best_model.predict_proba(self.X_holdOUT)[:, 1]]).ravel()
            print(f"{tn}\t{fp}\t{fn}\t{tp}")
            plot_roc_curves(model_labels, self.data_stats, PLOTS_PATH)
        else:
            cnf_matrix = confusion_matrix(self.Y_holdOUT, best_model.predict(self.X_holdOUT))
            plot_confusion_figure(cnf_matrix, best_model.model.classes_, PLOTS_PATH)
            fp = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
            fn = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
            tp = np.diag(cnf_matrix)
            tn = cnf_matrix.sum() - (fp + fn + tp)
            print(f"Confusion matrix:\n{cnf_matrix}\n")
            classes = list(set(self.Y_holdOUT))
            classes.sort()
            print(f"YP:{classes}")
            print(f"TP:{tp}")
            print(f"TN:{tn}")
            print(f"FP:{fp}")
            print(f"FN:{fn}\n\n")

            #plot_roc_curves(model_labels, self.data_stats, PLOTS_PATH)

        return best_model

       

    def main(self):
        print(f'{datetime.now()} Start fine-tuning of feature selection')

        self.feature_selection()

        print(f'{datetime.now()} Done\n' +
              f'\tSelected {len(self.selected_features)} of {self.X_visible.shape[1]} features.' +
              f'\n\t{self.selected_features}\n' +
              f'{datetime.now()} Start fine-tuning of Models')

        performances = self.fine_tune_models()

        print(f'{datetime.now()} End of fine-tuning\n' +
              f'{datetime.now()} Start of model selection')

        best_model = self.measure_and_select(performances)

        print(f'{datetime.now()} End of model selection' +
              f'{datetime.now()} Start of SHAP explainer')

        plot_shap_figure(best_model.model, self.X_holdOUT, PLOTS_PATH, self.binary_class)

        print(f'{datetime.now()} End of SHAP explainer' +
              f'{datetime.now()} Start of final model creation')

        """Before store the model we should train model over all data and we can store the model for further usage"""
        """Free memory"""
        del(self.Y_holdOUT)
        del(self.X_holdOUT)
        del(self.Y_visible)
        del(self.X_visible)


        """Load all data from csv file"""
        X, Y = DataLoading(data_path=DATA_PATH, verbose=self.verbose).load_dataset(splited=False)

        """Train model on all dataset with selected features, and store it"""
        best_model.fit(X[self.selected_features], Y)
        best_model.save_model()
        print(f'{datetime.now()} End')


if __name__ == "__main__":

    print('Starting of model creation')
    
    """Random Select NumberOfConfig from defined range of parameters via computation of all possible combinations and 
    selecting randomly defined number of configurations for each model"""
    NumberOfConfig = 50
    scoring_function = make_scorer(roc_auc_evaluation, greater_is_better=True)
    defined_configs = [ #XGBoost Classifier range of parameters
                        {'max_depth': [6, 7, 8, 9, 11, 13],
                        'learning_rate': [0.005, 0.01, 0.015],
                        'subsample': [0.65, 0.7, 0.75, 0.8, 0.85],
                        'colsample_bytree': [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
                        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                        'gamma': [0, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0],
                        'n_estimators': [1000, 1500, 2000, 2500, 3000],
                        'eval_metric': ['auc'],
                        'tree_method': ['gpu_hist'],
                        'predictor': ['gpu_predictor'],
                        'objective': ['binary:logistic'],
                        'use_label_encoder': [False]},

                        #RandomForest classifier range of parameters

                       {'n_jobs': [60],
                        'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13],
                        'max_features': ['sqrt'],
                        'min_samples_leaf': [1, 2, 3, 4, 5],
                        'min_samples_split': [2, 3, 4, 5, 6, 7, 8],
                        'n_estimators': [1000, 1500, 2000, 2500, 3000, 3500]}
                       ]

    Piepeline(FS=Lasso,
              Models=[XGBClassifier, RandomForestClassifier],
              fs_grid_params={'alpha': np.arange(0.00001, 0.0001, 0.00001)},
              Models_grid_params=defined_configs,
              NumberOfConfig=NumberOfConfig,
              scorer=scoring_function)
