# SAMLP: Semi-Automatic Machine Learning Pipeline.

Current repository provides implementation of Semi-Automatic Machine Learning Pipeline that was initially developed under 
the research project of Twitter Bot detection with title: "BotArtist: Generic approach for bot detection in
Twitter via semi-automatic machine learning pipeline".
Developed Machine Learning Pipeline allow to utilize without any significant adaptation, where only set of model 
hyperparameters require changes in order to avoid over/under-fitting due the solution complexity.

## Implementation

The SAMLP provide the entire machine learning pipeline, including:
 - Data train/test split
 - Feature selection
   - Base on the Lasso model
   - Including dynamic alpha hyperparameter searching
   - Class imbalance consideration during the feature selection
   - K-Fold Cross validation
 - Classification (binary/multiclass) hyperparameter fine-tuning
   - Models: XGBoost, RandomForest, GNB
   - Consideration of the class imbalance 
   - K-Fold Cross validation
   - F1 performance measurement due the class imbalance 
 - Decision thresholding fine-tuning (binary only)
 - Statistics of the model performances
 - Final model explainability
 - The resulting model is finally trained ove ALL data and stored as pkl file

<img src='plots/pipeline.png' width='450'>

## Requirements
Installation of required packages:
```bash
python3.9 -m pip install -r requirements.txt
```

For the utilization, please change the proper file names in the *configuration.py* file, providing the proper initial input filename, located in the *data/* folder .
```angular2html
├── README.md
├── configuration.py
├── pipeline.py
├── requirements.txt
└── utilities
    ├── DataLoading.py
    ├── feature_selector.py
    ├── model_selector.py
    ├── model_wrapper.py
    ├── mongoConfig.py
    ├── mongoConnector.py
    └── plotting.py
└── data (Required for data storage)
└── plots
└── stats (Required to store the statistics)
```
## Example of usage
Execution of the SAMLP is triggered by the pipeline.py python script. Script could parse multiple terminal arguments 
some of them is required for execution and some of them are optional. The tota list of arguments and their description:
 - [-f dataset.csv] filename of the dataset in CSV format. This dataset would be 
processed and further separated into train/validation and test portions (Required).
 - [-p /path/to/data/storage] path to the input file location. This path would also be used to store the separated 
data splits (Required).
 - [-s user_id,id,ID] comma separated names of the unique identifiers/unique per sample items columns that should 
be excluded from the dataset, if exist (Optional).
 - [-t target] name of the label/target column that should be utilized as Y value. The default values is 'target' (Optional).
 - [-o /path/to/output/folder] path to the output folder where results should be stored. In other case data would be 
stored in new directory based on the dataset filename (Optional).
Execution example:
 ```
python3 pipeline.py -f dataset.csv -p ~/my_data/ -s user_id,ID -t label -o ~/my_model_stats/
 ```
In case when you require to change model hyper-parameter ranges they could be found and updated in the configuration.py 
file. In this file it is possible to include and exclude the different models and update they lookup parameters ranges, 
including the number of testing configurations. 

Through the configuration file, it is possible to change the entire setup of your pipeline, including models, 
configurations, k-Fold cross validation, number of tested configurations and event starting point of the Lasso feature selection.

The only limitation of existing semi-automatic pipeline in the current version, in order to utilize existing models 
they should have implemented function adopting .fit .predict and .predict_proba . In the next version we will update 
functionality in order to support wide range of other ML models.

## Model Explainability
As model explainability we utilize well known SHAP game theoretical approach. Such methodology allow to provide important insights of the model captured difference between the sample classes (binary and multi-class). 

## Citation and contact:

TBD

