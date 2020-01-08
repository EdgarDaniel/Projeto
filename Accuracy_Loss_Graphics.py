
import numpy as np
import matplotlib as plt

def read_data(path_to_train_accuracy, path_to_train_loss,path_to_validation_accuracy,path_to_validation_loss):

    accuracy_train = np.genfromtxt(path_to_train_accuracy,delimiter=",")
    loss_train = np.genfromtxt(path_to_train_loss,delimiter=",")
    accuracy_validation = np.genfromtxt(path_to_validation_accuracy,delimiter=",")
    loss_validation = np.genfromtxt(path_to_validation_loss,delimiter=",")

    return accuracy_train,loss_train,accuracy_validation,loss_validation

def main():

    path_to_train_accuracy = 'accuracy_train_data.csv'
    path_to_train_loss = 'loss_train_data.csv'
    path_to_validation_accuracy = 'accuracy_validation_data.csv'
    path_to_validation_loss = '"loss_validation_data.csv"'


    # CREATE LIST OF NUMBER OF EPOCHS COMPUTED
    eval_indices = range(1, EPOCHS + 1)

    accuracy_train, loss_train, accuracy_validation, loss_validation = read_data(path_to_train_accuracy,path_to_train_loss,
                                                                                 path_to_validation_accuracy,path_to_validation_loss)

    print(eval_indices)
    print("Accuracy Train: ",accuracy_train)
    print("Loss Train: " ,loss_train)
    print("Accuracy Validation: ", accuracy_validation)
    print("Loss validation: ", loss_validation)


    # DRAW THE ACCURACY GRAPH FOR VALIDATION AND TRAIN
    plt.clf()
    plt.subplot(211)
    plt.plot(eval_indices, accuracy_train, 'k--', label='TREINO')
    plt.plot(eval_indices, accuracy_validation, 'g-x', label='VALIDAÇÃO')
    plt.legend(loc='upper right')
    plt.xlabel('Épocas')
    plt.ylabel('ACERTO')
    plt.grid(which='major', axis='both')

    # DRAW THE LOSS GRAPH FOR VALIDATION AND TRAIN

    plt.subplot(212)
    # plt.plot(eval_indices, train, 'g-x', label='Train Set Accuracy')
    plt.plot(eval_indices, loss_train, 'r-x', label='TREINO')
    # plt.plot(eval_indices, np.ones(len(eval_indices))/TOT_CLASSES, 'k--')
    plt.plot(eval_indices, loss_validation, 'k--', label='VALIDAÇÃO')
    plt.legend(loc="upper right")
    plt.xlabel('Épocas')
    plt.ylabel('ERRO')
    plt.ylim(0, 1)
    plt.grid(which='both', axis='y')

    plt.subplots_adjust(left=0.2, wspace=0.2, hspace=0.3)

    plt.show()
    plt.pause(0.01)
    plt.savefig('Learning.png')
