""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Parameter fine-tuning and Feature selection for ML model.
####################################################################################################################"""

import pandas as pd

from datetime import datetime

from utilities.DataLoading import DataLoading
from utilities.feature_selector import FeatureSelector
from utilities.model_selector import ModelSelector
from utilities.plotting import plot_shap_figure, plot_confusion_figure


class Piepeline:
    def __init__(self, stratified=True,
                       shuffle=True, verbose=True):
         
        self.verbose = verbose
        self.stratified = stratified
        self.shuffle = shuffle

        self.main()
       

    def main(self):

        """Loading the Data splited in train/test and hold-out portions"""
        DL = DataLoading(verbose=self.verbose)

        """Load only the visible data portion containing Train/Validation"""
        print(f'{datetime.now()} Pipeline: Data Loading\n')
        X_train, Y_train = DL.load_dataset(train=True, test=False, splited=True)

        """Create feature selector"""
        print(f'{datetime.now()} Feature selection')
        FS = FeatureSelector(stratified=self.stratified, shuffle=self.shuffle, verbose=self.verbose)

        """Get best features based on the visible data portion, with use of Lasso feature selector"""
        selected_features = FS.get_features(X_train, Y_train)

        print(f'{datetime.now()} End of fine-tuning\n' +
              f'{datetime.now()} Start of model fine-tuning')


        """Drop noisy/un-selected features"""
        X_train.drop([feature for feature in X_train.columns if feature not in selected_features],
                     axis=1, inplace=True)

        MS = ModelSelector(Y_train, stratified=self.stratified, shuffle=self.shuffle, verbose=self.verbose)
        MS.fine_tune_models(X_train, Y_train)

        print(f'{datetime.now()} End of fine-tuning\n' +
              f'{datetime.now()} Start of testing')

        X_hold, Y_hold = DL.load_dataset(train=False, test=True, splited=True)
        """Drop noisy/un-selected features"""
        X_hold.drop([feature for feature in X_hold.columns if feature not in selected_features],
                     axis=1, inplace=True)


        MS.models[MS.best_model_index]
        MS.measure_hold_out(X_hold, Y_hold)

        print(f'{datetime.now()} Shap explain plotting')
        """Plot SHAP explanability"""
        plot_shap_figure(MS.models[MS.best_model_index], X_hold, binary=MS.binary_class)
        
        
        plot_confusion_figure(MS.models[MS.best_model_index], X_hold, Y_hold)
        """Merge train and test dataset, train the final model and store it"""
        MS.store_final_model(pd.concat([X_train, X_hold]), pd.concat([Y_train, Y_hold]))

        print(f'{datetime.now()} End')


if __name__ == "__main__":

    print('Starting of model creation')

    _ = Piepeline()
