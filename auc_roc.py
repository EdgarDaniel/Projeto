
##COPYRIGHT EDGAR DANIEL

#THIS SCRIPT CALCULATES THE AUC VALUE AND DRAWS THE ROC CURVE GRAPHIC FORM THE LABELS AND
#RESULTS SAVED IN THE CSV FILES WHEN USED THE VALIDATION DATA


# NUMPY AND MATPLOTLIB IMPORT
import numpy as np
import matplotlib.pyplot as plt
# ROC CURVE AND AUC SCORE IMPORT
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

#FUNCTION TO WRITE THE AUC VALUE TO A TXT FILE NAMED "AUC_VALUE.txt"
def write_auc_value(auc_value):
    f = open("AUC_VALUE.txt","w")
    f.write(str(auc_value))
    f.close()

#FUNCTION THAT READ DATA FROM THE CSV FILES
#INPUT: path_to_lables -> PATH TO THE FILE THAT STORES THE LABELS
#     : path_to_results -> PATH TO THE FILE THAT STORES THE RESULTS
def read_data(path_to_labels, path_to_results):

    results = np.genfromtxt(path_to_results,delimiter=",")
    labels = np.genfromtxt(path_to_labels,delimiter=",")

    return results,labels

#FUNCTION THAT DRAW THE ROC GRAPHIC
#INPUT: fpr -> RECEIVE THE FALSE POSITIVE RATE FROM THE ROC_CURVE FUNCTION
#     : tpr -> RECEIVE THE TRUE POSITIVE RATE FROM THE ROC_CURVE FUNCTION
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    #SAVES THE GRAPHIC IN THE FILE "ROC_CURVE.png"
    plt.savefig("ROC_CURVE.png")
    plt.show()


def main():

    #PATH TO THE FILES WITH THE LABELS AND PROBABILIST RESULTS FROM NETWORK
    path_labels= '/home/edgardaniel/Projeto/labelsOF.csv'
    path_results = '/home/edgardaniel/Projeto/resultsOF.csv'

    #READ DATA FROM FILES CSV
    results, labels = read_data(path_labels,path_results)

    #USE ONLY PROBABILIST FOR 1 CLASS
    resultspro = results[:,1]

    #VARIABLES THAT CONTROL THE SIZE AND THE ARRAYS
    print(resultspro)
    print(resultspro.size)
    print(labels)
    print(labels.size)

    #CALCULATE THE NUMBER FOR AUC
    auc = roc_auc_score(labels,resultspro)
    print("AUC VALUE", auc)
    write_auc_value(auc)

    # CALCULATE THE TRUE POSITIVE AND FALSE POSITIVE AND THRESHOLDS FOR DRAW GRAPHIC
    fpr, tpr, thresholds = roc_curve(labels,resultspro)

    # DRAW THE AUC GRAPHIC
    plot_roc_curve(fpr, tpr)

#CALLS THE MAIN FUNCTION
main()
