import argparse
import pandas as pd

from utilities.DataLoading import DataLoading
from utilities.model_wrapper import ModelWrapper
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('-i', dest="inputfilename", required=True,
                    help='Path to the input file to predict by model.')
parser.add_argument('-o', dest="outputpath", default=None,
                    help='The directory where the results would be stored. If not provided, script will create folder based on filename')
parser.add_argument('-t', dest="target", default="target",
                    help='Name of the target column (default: "target") example: -t class_target ')
if __name__ == '__main__':
    args = parser.parse_args()
    model = ModelWrapper(output_path=args.outputpath)
    model.load_model()

    X = pd.read_csv(args.inputfilename)
    Y = X['target']

    X_F = X[model.features]

    pred_Y = model.predict(X_F)
    print(pred_Y)
    print(Y)
    print( confusion_matrix(Y, pred_Y))
    print(X[pred_Y == 1][['Domain','IP', 'Port']])

    #ROC_AUC = roc_auc_score(Y, pred_Y, multi_class="raise")
    F1 = f1_score(Y, pred_Y, average='binary')
    precision = precision_score(Y, pred_Y, average='binary')
    recall = recall_score(Y, pred_Y, average='binary')
    report = classification_report(Y, pred_Y)

    logs = f"Model achieve scores over test set: F1:{F1} Precision:{precision} Recall:{recall}\n"

    tn, fp, fn, tp = confusion_matrix(Y, pred_Y).ravel()
    logs += f"\nTN\t\tFP\t\tFN\t\tTP\n{tn}\t{fp}\t{fn}\t{tp}\n"

    print(logs)
