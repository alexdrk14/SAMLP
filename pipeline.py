""""####################################################################################################################
Author: TBD
E-mail: TBD
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
                       outputpath, extra_tests,
                       stratified=True,
                       shuffle=True, verbose=True):
         
        self.verbose = verbose
        self.stratified = stratified
        self.shuffle = shuffle

        self.filename = filename
        if not datapath.endswith('/'):
            datapath += '/'
        self.datapath = datapath
        self.extra_tests = extra_tests.strip().split(',') if len(extra_tests) > 0 else None

        outputpath = outputpath if outputpath is not None else f'./{filename.split(".csv")[0]}/'

        if not outputpath.endswith('/'):
            outputpath += '/'
        self.outputpath = outputpath

        """Check if all necessary folders are exist or need to be created"""
        if not os.path.isdir(self.outputpath):
            os.mkdir(self.outputpath)
            os.mkdir(self.outputpath + "stats/")
            os.mkdir(self.outputpath + "plots/")


        self.main(sensitive, target)
       

    def main(self, sensitive, target):

        """Loading the Data splited in train/test and hold-out portions"""
        DL = DataLoading(filename=self.filename, data_path=self.datapath,
                        sensitive=sensitive, target=target,
                        verbose=self.verbose)

        """Load only the visible data portion containing Train/Validation"""
        print(f'{datetime.now()} Pipeline: Data Loading\n')
        X_train, Y_train = DL.load_dataset(train=True, test=False, splited=True)

        """Create feature selector"""
        print(f'{datetime.now()} Feature selection')
        FS = FeatureSelector(output_path=self.outputpath,
                             stratified=self.stratified, shuffle=self.shuffle,
                             verbose=self.verbose)

        """Get best features based on the visible data portion, with use of Lasso feature selector"""
        selected_features = FS.get_features(X_train, Y_train)

        print(f'{datetime.now()} End of fine-tuning\n' +
              f'{datetime.now()} Start of model fine-tuning')


        """Drop noisy/un-selected features"""
        X_train.drop([feature for feature in X_train.columns if feature not in selected_features],
                     axis=1, inplace=True)

        MS = ModelSelector(Y_train, stratified=self.stratified, 
                           shuffle=self.shuffle, 
                           verbose=self.verbose, 
                           output_path=self.outputpath)

        MS.fine_tune_models(X_train, Y_train)

        print(f'{datetime.now()} End of fine-tuning\n' +
              f'{datetime.now()} Start of testing')

        X_hold, Y_hold = DL.load_dataset(train=False, test=True, splited=True)
        """Drop noisy/un-selected features"""
        X_hold.drop([feature for feature in X_hold.columns if feature not in selected_features],
                     axis=1, inplace=True)

        MS.models[MS.best_model_index]
        MS.measure_hold_out(X_hold, Y_hold)

        if self.extra_tests is not None:
            logs = ''
            for dataset_name in self.extra_tests:
                logs += f'{dataset_name} : '
                exX_hold, exY_hold = DL.load_dataset(splited=False, filename=dataset_name)
                """Drop noisy/un-selected features"""
                exX_hold.drop([feature for feature in X_hold.columns if feature not in selected_features],
                            axis=1, inplace=True)
                logs += MS.measure_hold_out(exX_hold, exY_hold, raw=True) + '\n'
            logs += f'Total : ' + MS.measure_hold_out(X_hold, Y_hold, raw=True) + '\n'
            f_out = open(f'{self.outputpath}multi_test_logs.txt', 'w+')
            f_out.write(logs)
            f_out.close()


        print(f'{datetime.now()} Shap explain plotting')
        """Plot SHAP explanability"""
        try:
            plot_shap_figure(MS.models[MS.best_model_index], X_hold, binary=MS.binary_class)
        except Exception as e:
            print(f'SHAP error: {e}')
            print('skiping the shap .... ')
        
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
parser.add_argument('-o', dest="outputpath", default=None,
                    help='The directory where the results would be stored. If not provided, script will create folder based on filename')

parser.add_argument('--tests', dest="extratests", default='',
                    help='Comma separated filenames that also should be used for model testing:--tests  my_new_test.csv,extra_test_file.csv')
if __name__ == "__main__":
    args = parser.parse_args()

    print(f'Starting of model creation with filename: {args.filename} and data path:{args.datapath}')
    sensitive = args.sensitive.split(',') if args.sensitive != "" else None
    print(f'\tSensitive columns: {sensitive}')
    print(f'\tTarget: {args.target}')
    _ = Piepeline(filename=args.filename,
                  datapath=args.datapath,
                  outputpath=args.outputpath,
                  sensitive=sensitive,
                  target=args.target,
                  extra_tests = args.extratests)
