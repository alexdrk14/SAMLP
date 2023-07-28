""""####################################################################################################################
Author: Alexander Shevtsov ICS-FORTH
E-mail: shevtsov@ics.forth.gr
-----------------------------------
Dynamic Model class that can take any possible model with limitation of fit/predict functions  and allow:
    - random select possible set of hyper-parameter configurations
    - train/prediction/proba predictions
    - load / store of trained model
    - loading of fine-tuned threshold for better prediction
####################################################################################################################"""
import random, ast, pickle, itertools
import pandas as pd

"""
   Model class that store the entire model and configurable set of parameters, 
   used in order to reduce complexity of fine-tuning method
"""
STATS_PATH = 'stats/'


class Model:

    def __init__(self, nmbr_to_select=0, feature_category="", configs_ranges={}, model=None):

        self.__model_origin = model
        self.scaller = None
        self.feature_category = feature_category
        self.decision_th = None
        self.features = None

        """Define number of configurations that should be selected from 
        pre-define spaces of possible hyper-parameter configurations"""
        if nmbr_to_select > 0:
            self.create_parameters_list(nmbr_to_select, configs_ranges)

    """Create list of hyper-parameters for model via random selection from pre-defined possible hyper-parameter range"""
    def create_parameters_list(self, select, dict_range):
        config_keys = list(dict_range.keys())

        conf = [dict_range[param] for param in config_keys]

        selected = random.sample(list(itertools.product(*conf)), select)
        self.parameters = [{config_keys[i]: sample[i] for i in range(len(config_keys))} for sample in selected]

    """Create model based on specific parameters"""
    def create_model(self, params):
        """Store current model configuration"""
        self.config = ast.literal_eval(params) if type(params) == str else params

        """Parse parameters to original model"""
        self.model = self.__model_origin(**self.config)

    def load_params(self):
        """Load selected parameters from fine-tuned model for particular
        feature category and create original model based on those parameters"""
        self.create_model(ast.literal_eval(open("stats/best_model_params.txt", "r").read().split("\n")[0]))

    def store_params(self):
        """Store selected parameters"""
        f_out = open(f'{STATS_PATH}best_model_params.txt', "w+")
        f_out.write(f'{self.config}\n')

    """Store model in form of pickle object for further usage"""
    def save_model(self):
        self.store_params()
        pickle.dump(self.model, open(f'{STATS_PATH}model.pkl', "wb"))


    """Read pickle form of pre-trained model for further usage as predictor"""
    def load_model(self):
        df = pd.read_csv(f'{STATS_PATH}pipeline_result.csv', sep='\t')
        best_model = df['valid_rocauc'].idxmax()
        
        self.decision_th = df.iloc[best_model]['decision_threshold'] if df.iloc[best_model]['decision_threshold'] > 0.0 else None
        self.features = ast.literal_eval(df.iloc[best_model]['features'])

        with open(f'{STATS_PATH}model.pkl', "rb") as f_in:
            self.model = pickle.load(f_in)

    def fit(self, x, y):
        self.model.fit(x, y)

    def train_predict(self, x_train, y_train, x_val, probs=True):
        self.fit(x_train, y_train)
        return (self.predict_proba(x_train), self.predict_proba(x_val)) if probs else (self.predict(x_train), self.predict(x_val))

    """Prediction function that predict without decision correction in case of no proper threshold
    In case of correction threshold (after fine-tuning) we predict with proper prediction threshold"""
    def predict(self, X):
        if self.features is not None:
            X = X[self.features]

        if self.decision_th is None:
            return self.model.predict(X)
        else:
            Probs = self.predict_proba(X)[:, 1].copy()
            return Probs > self.decision_th

    def predict_proba(self, X):
        if self.features is not None:
            X = X[self.features]
        return self.model.predict_proba(X)



