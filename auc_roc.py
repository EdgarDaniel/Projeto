
import numpy as np
import matplotlib.pyplot as plt
# ROC CURVE AND AUC SCORE
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def read_data(path_to_labels, path_to_results):

    results = np.genfromtxt(path_to_results,delimiter=",")
    labels = np.genfromtxt(path_to_labels,delimiter=",")

    return results,labels

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig("ROC_CURVE.png")
    plt.show()


def main():

    path_labels= '/home/edgardaniel/Projeto/labelsOF.csv'
    path_results = '/home/edgardaniel/Projeto/resultsOF.csv'

    results, labels = read_data(path_labels,path_results)

    auc = roc_auc_score(labels.ravel(),results.ravel())
    print("AUC VALUE", auc)

    # CALCULATE THE AUC VALUES
    fpr, tpr, thresholds = roc_curve(labels.ravel(),results.ravel())

    # DRAW THE AUC GRAPHIC
    plot_roc_curve(fpr, tpr)

main()
