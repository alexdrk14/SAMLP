""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr, shevtsov@csd.uoc.gr
-----------------------------------
Parameter fine-tuning and Feature selection for ML model.
####################################################################################################################"""

import pandas as pd
import argparse, os

from datetime import datetime

import configuration as cnf
from utilities.DataLoading import DataLoading
from utilities.feature_selector import FeatureSelector
from utilities.model_selector import ModelSelector
from utilities.plotting import plot_shap_figure, plot_confusion_figure


class Piepeline:
    def __init__(self, filename, datapath,
                       sensitive, target,
                       stratified=True,
                       shuffle=True, verbose=True):
         
        self.verbose = verbose
        self.stratified = stratified
        self.shuffle = shuffle

        self.filename = filename
        if not datapath.endswith('/'):
            datapath += '/'
        self.datapath = datapath

        """Check if all necessary folders are exist or need to be created"""
        if os.path.isdir(cnf.PLOTS_PATH):
            os.mkdir(cnf.PLOTS_PATH)
        if os.path.isdir(cnf.STATS_PATH):
            os.mkdir(cnf.STATS_PATH)

        self.main(sensitive, target)
       

    def main(self, sensitive, target):

        """Loading the Data splited in train/test and hold-out portions"""
        DL = DataLoading(filename=self.filename, data_path=self.datapath,
                        sensitive=sensitive, targe=target,
                        verbose=self.verbose)

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


parser = argparse.ArgumentParser()
parser.add_argument('-f', dest="filename", required=True,
                    help='Name of the input file , that would be split into train/test portions and user for the model creation')
parser.add_argument('-p', dest="datapath", required=True,
                    help='data folder path')
parser.add_argument('-s', dest="sensitive", default="",
                    help='Comma separated names of the columns that should be excluded from the training/testing data example:-s user_id,id,ID ')
parser.add_argument('-t', dest="target", default="target",
                    help='Name of the target column (default: "target") example: -t class_target ')
if __name__ == "__main__":
    args = parser.parse_args()

    print(f'Starting of model creation with filename: {args.filename} and data path:{args.datapath}')
    sensitive = args.sensitive.split(',') if args.sensitive != "" else None
    print(f'\tSensitive columns: {sensitive}')
    print(f'\tTarget: {args.target}')
    _ = Piepeline(stratified=args.filename,
                  datapath=args.datapath,
                  sensitive=sensitive,
                  target=args.target)
