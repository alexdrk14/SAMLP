""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Parameter fine-tuning and Feature selection for ML model.
####################################################################################################################"""

import numpy as np
import pandas as pd
import configuration as cnf


from tqdm import tqdm
from operator import itemgetter
from utilities.model_wrapper import ModelWrapper
from collections import defaultdict


"""Models that we use"""
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve, classification_report


"""Additional function which compute best decision threshold in order to maximize the PrecisionVSRecall or ROC.
Return the found best decision threshold and corrected predicted targets."""
def compute_best_threshold(Y_true, Y_predicted, comp_type="PR"):
    if comp_type =="PR":
        """Compute the Precision vs Recall Curve"""
        precision, recall, thresholds = precision_recall_curve(Y_true, Y_predicted)

        """Get f-score for each threshold"""
        fscore = (2 * precision * recall) / (precision + recall)

        """Identify the threshold with the highest performance"""
        best_threshold = thresholds[np.argmax(fscore)]

    else:
        """In case maximize the ROC-AUC we should use roc_curve"""
        """Compute and store figure with ROC-AUC performance"""
        fpr, tpr, th = roc_curve(Y_true, Y_predicted)

        """calculate the g-mean for each threshold"""
        g_mean = np.sqrt(tpr * (1 - fpr))

        """locate the index of the larger g-mean"""
        ix = np.argmax(g_mean)

        best_threshold = th[ix]

    return best_threshold, Y_predicted >= best_threshold


class ModelSelector:

    def __init__(self, Y, stratified=True, shuffle=True, verbose=True):


        """Local variables that used in order to identify if the dataset is for binary classification or for regression"""
        self.binary_class = True if len(set(Y)) == 2 else False
        self.cont_values = True if type(list(Y)[0]) == np.float64 or len(
            set(Y)) > 15 else False



        """In case of binary classification:
        Check if dataset is imbalance, in case of imbalance we should fix class weights in models"""
        if self.binary_class and len(set(Y)) == 2 and sum(Y == 0) != sum(Y == 1):
            """Since our dataset is imbalanced in terms of targets we need to compute proper weights per outcome for each model"""

            neg_count = sum(Y == 0)
            pos_count = sum(Y == 1)

            for model_index in cnf.utilize_models:
                if cnf.Models[model_index] == XGBClassifier:
                    cnf.Models_grid_params[model_index]['scale_pos_weight'] = [neg_count / pos_count]
                elif cnf.Models[model_index] == RandomForestClassifier or cnf.Models[model_index] == SVC:
                    cnf.Models_grid_params[model_index]['class_weight'] = [{
                        0: (Y.shape[0] / (2 * neg_count)),
                        1: (Y.shape[0] / (2 * pos_count))
                    }]
        elif len(set(Y)) != 2:
            for model_index in cnf.utilize_models:
                if cnf.Models[model_index] == XGBClassifier:
                    cnf.Models_grid_params[model_index]['objective'] = ['multi: softprob']
                    cnf.Models_grid_params[model_index]['num_class'] = [len(set(Y))]

        self.models = [ModelWrapper(nmbr_to_select=cnf.NumberOfConfig, configs_ranges=cnf.Models_grid_params[i], model=cnf.Models[i]) for i
                       in cnf.utilize_models]


        self.stratified = stratified
        self.shuffle = shuffle
        self.verbose = verbose



    """At the end update the class variable self.models, where only est configurational models are kept after 
    hyper-parameter fine-tuning with use of K-Fold Cross validation.
    Final models are trained over the train and validation dataset portions.
    --------------------------------------------------------------------------------------------------------------------
    NOTE: The X data portion should be parsed into this function with already selected features. 
    This function do not utilize any feature selection methods and ONLY concentrates over models fine-tuning
    --------------------------------------------------------------------------------------------------------------------
    ALSO: The entire X,Y data will be used for hyper-parameter fine-tuning and the best models will be trained over the 
    entire dataset at the end. For this reason:
     -------IT'S IMPORTANT TO SPLIT AND HIDE HOLD-OUT DATASET BEFORE CALLING THIS FUNCTION--------"""
    def fine_tune_models(self, X, Y):
        """Dictionary for storing performances (train, validation) for each model, configuration during the K-Fold cross val."""
        performances = defaultdict(lambda: defaultdict(lambda: {"perf-val": [], "perf-train": []}))

        """K-FOLD CrossValidation"""
        fold_ind = 0
        skfold = StratifiedKFold(n_splits=cnf.FOLD_K, shuffle=self.shuffle)
        for train_index, val_index in skfold.split(X, Y):
            """Split data into Train folds and Validation Folds"""
            train_X, val_X = X.iloc[train_index, :], X.iloc[val_index, :]
            train_Y, val_Y = Y.iloc[train_index,], Y.iloc[val_index,]

            print(f'Fold index:{fold_ind + 1} of {cnf.FOLD_K}')

            """For each model and its own selected configuration train the model over TRAIN data portion and measure Validation performance"""
            for model_index, (model) in enumerate(self.models):
                print(f'Model: {model_index + 1}')
                for model_config in tqdm(model.parameters):
                    model.create_model(model_config)

                    """Train model in train data, predict train data and predict validation data"""
                    YP_train, YP_val = model.train_predict(train_X, train_Y, val_X, probs=False)

                    """In case of binary classification we can improve validation performance by tuning the decision threshold"""
                    #if self.binary_class:
                    #    """Find best separation threshold and correct the predictions """
                    #   best_threshold, YP_val = compute_best_threshold(val_Y,
                    #                                                    model.predict_proba(val_X)[:, 1],
                    #                                                    comp_type="PR")

                    performances[model_index][f'{model_config}']["perf-val"].append(f1_score(val_Y, YP_val, average='macro' if not self.binary_class else 'binary'))
                    performances[model_index][f'{model_config}']["perf-train"].append(f1_score(train_Y, YP_train, average='macro' if not self.binary_class else 'binary'))

                    """Keep false positive rate, true positive rate for later plotting with shadowing for ROC-AUC curves"""
                    # fpr, tpr, _ = roc_curve(val_Y, YP_val)
                    # precision, recall, _ = precision_recall_curve(val_Y, YP_val)

                    # self.data_stats[i][fold_ind][f'{model_config}']["FP"] = fpr
                    # self.data_stats[i][fold_ind][f'{model_config}']["TP"] = tpr

            fold_ind += 1

        """List of tuples where for each model we store the best configuration and avg performances"""
        self.best_model_config = []

        """Compute averages performances over each model and configuration during the k-fold cross validation. 
        After we store for each model single configuration and the avg performances (train , validation)"""
        for model_index in range(len(self.models)):
            for config in performances[model_index]:
                for metric in performances[model_index][config]:
                    performances[model_index][config][metric] = sum(performances[model_index][config][metric]) / len(
                        performances[model_index][config][metric])

            best_config = [(config, performances[model_index][config]["perf-val"]) for config in
                                 performances[model_index]]
            
            best_config, val_performance = max(best_config, key=itemgetter(1))
            

            """Store best configuration in form of string, avg train performance, avg validation performance as tuple"""
            self.best_model_config.append( (best_config, performances[model_index][best_config]['perf-train'], val_performance) )

            """Keep the final models trained on train and validation data portions"""
            self.models[model_index].create_model(best_config)
            self.models[model_index].fit(X, Y)

        """Get best configuration tuples based on avg validation performance"""
        

        the_One_best = max(self.best_model_config, key=lambda data:data[2])
        self.best_model_index = self.best_model_config.index(the_One_best)


    def measure_hold_out(self, X_hold_out, Y_hold_out):
        pipeline_result = []

        for model_index in range(len(self.models)):

            config, train_perf, val_perf = self.best_model_config[model_index]

            Y_pred = self.models[model_index].predict(X_hold_out) if not self.binary_class else self.models[model_index].predict_proba(X_hold_out)
            best_threshold = None

            """In case of the binary classification compute best decision threshold and update the predicted targets"""
            if self.binary_class:
                Y_pred = Y_pred[:, 1]

                """Find best separation threshold and correct the predictions """
                best_threshold, Y_pred = compute_best_threshold(Y_hold_out,
                                                                Y_pred,
                                                                comp_type="PR")

            hold_out_perf = f1_score(Y_hold_out, Y_pred, average='macro' if not self.binary_class else 'binary')  # roc_auc_evaluation(self.Y_holdOUT,model.predict(self.X_holdOUT))

            """Update the model decision threshold"""
            self.models[model_index].decision_th = best_threshold

            pipeline_result.append([cnf.Models_Names[model_index],
                                    train_perf, val_perf, hold_out_perf,
                                    best_threshold, config])

        """Store the logs of each best configurations per models with all computed stats"""
        pipeline_result = pd.DataFrame(pipeline_result,
                                       columns=['name', 'train_perf', 'valid_perf', 'holdout_perf',
                                               'decision_threshold', 'params'])

        pipeline_result .to_csv(f'{cnf.STATS_PATH}pipeline_result.csv', index=False, sep='\t')

        
        Y_pred = self.models[self.best_model_index].predict_proba(X_hold_out)
        if self.binary_class: 
            Y_pred = Y_pred[:, 1] >= self.models[self.best_model_index].decision_th

        ROC_AUC = roc_auc_score(Y_hold_out, Y_pred, multi_class="raise" if self.binary_class else 'ovr')
        
        if not self.binary_class:
            Y_pred = self.models[self.best_model_index].predict(X_hold_out)

        F1 = f1_score(Y_hold_out, Y_pred, 
                      average='binary' if self.binary_class else 'macro')

        precision = precision_score(Y_hold_out, Y_pred,
                          average='binary' if self.binary_class else 'macro')

        recall = recall_score(Y_hold_out, Y_pred,
                          average='binary' if self.binary_class else 'macro')

        if not self.cont_values and (cnf.MultyClassNames is None or len(cnf.MultyClassNames) < len(set(Y_hold_out))):
            cnf.MultyClassNames = [f"class {i}" for i in range(len(set(Y_hold_out)))]

        report = classification_report(Y_hold_out, Y_pred, target_names=[cnf.MultyClassNames[i] for i in self.models[self.best_model_index].model.classes_] )

        logs = f"Model achieve scores over test set: ROC-AUC:{ROC_AUC}, F1:{F1} Precision:{precision} Recall:{recall}\n" + \
               f"Classification report:\n{report}"
        if self.binary_class:
            tn, fp, fn, tp = confusion_matrix(Y_hold_out,Y_pred).ravel()
            logs += f"\nTN\t\tFP\t\tFN\t\tTP\n{tn}\t{fp}\t{fn}\t{tp}\n"

        print(logs)
        f_out = open(f"{cnf.STATS_PATH}report.txt", "w+")
        f_out.write(f"{logs}")
        f_out.close()


    def store_final_model(self, X_ALL, Y_ALL):
        self.models[self.best_model_index].fit(X_ALL, Y_ALL)
        self.models[self.best_model_index].save_model()
        del(self.models)


