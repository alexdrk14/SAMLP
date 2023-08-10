import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import configuration as cnf

def plot_roc_curves(model_names, data, output_path):
    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize=(6, 6), dpi=600)

    for model_index in range(len(model_names)):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold in range(len(data[model_index])):
            # ------------------ Load parameters needed for plotting purposes ----------
            fpr = data[model_index][fold]["FP"]
            tpr = data[model_index][fold]["TP"]

            # -------------- Compute ROC-AUC --------------
            aucs.append(auc(fpr, tpr))
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # ----------------------- mean ROC curves --------------------------------------
        std_auc = np.std(aucs)

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        mean_fpr1 = mean_fpr
        mean_auc1 = auc(mean_fpr1, mean_tpr)

        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        collors = [4, 0, 1, 5, 3, 2]
        col = collors[model_index]

        plt.plot(mean_fpr1, mean_tpr, color='C{}'.format(col),
                 label=f'{model_names[model_index]} (AUC: {mean_auc1:.2f} $\pm$ {std_auc:.3f}',
                 lw=2, alpha=.8)

        plt.fill_between(mean_fpr1, tprs_lower, tprs_upper, color='C{}'.format(col), alpha=.1)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive rate', fontsize=13)
    plt.ylabel('True Positive rate ', fontsize=13)
    plt.title('ROC curves ', fontsize=15)
    plt.legend(loc='best')

    plt.savefig(f'{output_path}roc_curves.png', bbox_inches='tight', dpi=600, facecolor='w')


def plot_shap_figure(model_wrapper, data, binary=True):
    explainer = shap.TreeExplainer(model_wrapper.model)
    fig = plt.figure()
    if binary:
        shap_values = explainer(data)
        shap.summary_plot(shap_values, plot_type='violin', show=False)
        fig.savefig(f'{cnf.PLOTS_PATH}shap.png', bbox_inches='tight', dpi=600, facecolor='w')
        plt.clf()
    else:
        shap_values = explainer.shap_values(data)
        for categ in range(0,len(shap_values)):
            shap.summary_plot(shap_values[categ], feature_names=data.columns, plot_type='violin', show=False)
            fig.savefig(f'{cnf.PLOTS_PATH}shap_class_{categ}.png', bbox_inches='tight', dpi=600, facecolor='w')
            plt.clf()
        shap.summary_plot(shap_values, 
                          plot_type='bar', 
                          class_names=[cnf.MultyClassNames[ind] for ind in model_wrapper.model.classes_],
                          feature_names=data.columns,
                          show=False)
        fig.savefig(f'{cnf.PLOTS_PATH}shap_bar_all.png', bbox_inches='tight', dpi=600, facecolor='w')
        plt.clf()

def plot_confusion_figure(model_wrapper, data_X, data_Y):
    fig = plt.figure()

    YP = model_wrapper.predict(data_X) if model_wrapper.decision_th is None else model_wrapper.predict_proba(data_X)[:, 1] >= model_wrapper.decision_th
    #cm = confusion_matrix(data_Y, YP)
    
    ConfusionMatrixDisplay.from_predictions(data_Y, YP,
                                            display_labels=[cnf.MultyClassNames[ind] for ind in model_wrapper.model.classes_],
                                            xticks_rotation="vertical",
                                            colorbar=False)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                           display_labels=cnf.MultyClassNames)
    #disp.plot()
 
    plt.savefig(f'{cnf.PLOTS_PATH}confusion_matrix.png', bbox_inches='tight', dpi=600, facecolor='w')
    plt.clf()

