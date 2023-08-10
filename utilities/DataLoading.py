""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Implementation of data loading utility. It allows to load data from data path and manage to :
    - split them into train/validation (visible) and test (hold-out) portions
    - in case of repetitions script keeps original data split in order to keep testing set un-seen by model
    - also stability of test set and train/validation allow to multiple executions over different algorithms
      and keep their performances equal
####################################################################################################################"""

import pandas as pd
import numpy as np
from os import path
import utilities.mongoConfig as cnf
from sklearn.model_selection import train_test_split
import configuration as cnf


class DataLoading:

    def __init__(self, test_size=0.2, verbose=False, shuffle=True):
        """Define all type of features filenames (output files from feature extraction methods)"""

        data_path = cnf.DATA_PATH
        if not data_path.endswith('/'):
            data_path += '/'
        
        self.single_file = f'{cnf.DATA_PATH}{cnf.InputFileName}'

        self.visible_postfix = "_visible.csv"
        self.hidden_postfix = "_hold_out.csv"

        self.test_size = test_size
        self.verbose = verbose
        self.shuffle = shuffle

        self.main()

    """Check if data is already aligned or required pre-processing
        Return values: 0 (Don't require) 
                       1 (Required for all files)
    """
    def require_processing(self):
        to_check = [self.single_file.replace(".csv", self.visible_postfix), self.single_file.replace(".csv", self.hidden_postfix)]
        if False in [path.isfile(file) for file in to_check]:
            if self.verbose: print("-" * 5 + ': Required pre-processing')
            return True

        if self.verbose: print("-" * 5 + ': Pre-processing is not required')
        return False

    def prepare(self):
        if self.verbose: print("Data Loading: Prepare function")

        """Read CSV file and balance the classes"""
        df = pd.read_csv(self.single_file, header=0, sep=',')

        """Split the data into train/val and testing sets"""
        visible_df, test_df = self.split(df)

        """Store the Train/Test and Hold-out data portions in separate CSV files"""
        self.store_df(visible_df, test_df)


    """
        Split Dataframe into two portions with *_visible.csv (train/val) and *_hidden.csv (test) postfix.
        This procedure is performed for each feature categories since 
        different model type require different category of features.
    """
    def split(self, data):


        if self.verbose: print('Data Loading: Split')

        """ Stratified split dataset into two portions
            Visible (Train/Validation) and Test portion as a hold-out
        """
        X_visible, X_test, _, _ = train_test_split(data, data["target"],
                                                   test_size=self.test_size,
                                                   shuffle=self.shuffle,
                                                   stratify=data["target"])
        return X_visible, X_test


    def store_df(self, X_visible, X_test):
        """Store dataframes in separate files for each feature category as visible and hidden portion"""
        X_visible.to_csv(self.single_file.replace(".csv", self.visible_postfix),
                        sep=",", index=False)
        X_test.to_csv(self.single_file.replace(".csv", self.hidden_postfix),
                        sep=",", index=False)


    def _load_csv_file(self, filename):
        X = pd.read_csv(filename, sep=",", header=0)
        Y = X["target"].copy()
        to_drop = ['target']
        if 'IP' in X.columns:
            to_drop.append("IP")

        X.drop(to_drop, axis=1, inplace=True)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        return X, Y

    def load_dataset(self, train=False, test=False, splited=True):

        if self.verbose:
            print(f'Data Loading: read csv')
        if splited:
            if not train and not test:
                return None

            """Loading visible and hidden dataframes"""
            if train:
                X_train, Y_train = self._load_csv_file(self.single_file.replace(".csv", self.visible_postfix))
                Y_train = Y_train.astype(int)

            if test:
                X_test, Y_test = self._load_csv_file(self.single_file.replace(".csv", self.hidden_postfix))
                Y_test = Y_test.astype(int)
            targets = set(Y_train) if train else set(Y_test)

            if self.verbose:
                stats_v = []
                stats_h = []
                for target in targets:
                    if train:
                        stats_v.append(f"class {target}: {sum(Y_train == target)}")
                    if test:
                        stats_h.append(f"class {target}: {sum(Y_test == target)}")

                print('Data Loading: Loaded dataset with:' +
                      f'\n\t' + ' and '.join(stats_v) +
                      f'\n\t' + ' and '.join(stats_h))
            if train and test:
                return X_train, Y_train, X_test, Y_test
            if train:
                return X_train, Y_train
            if test:
                return X_test, Y_test
        else:
            X, Y = self._load_csv_file(self.single_file)
            Y = Y.astype(int)
            if self.verbose:
                stats = []
                for target in set(Y):
                    stats.append(f"class {target}: {sum(Y == target)}")
                print('Data Loading: Loaded dataset with:\n\t' +
                      ' and '.join(stats))
            return X, Y


    def main(self):
        if self.verbose:
            print("Data Loading: initialization")

        if self.require_processing():

            self.prepare()
            if self.verbose:
                print("\tComplete")





