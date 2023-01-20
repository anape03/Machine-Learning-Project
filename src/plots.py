import numpy as np
import matplotlib.pyplot as plt

def plotAllModels():
    """
    Create a plot using all model data (epochs, learning rate, neurons in hidden layer) 
    calculated from testing parameters: learning rate, neurons in hidden layer.
    (100 models)
    """
    list_epochs, list_alpha, list_M = np.loadtxt('all_data.csv', delimiter=',')

    colors = plt.cm.jet(np.linspace(0,1,10))

    M_value = 2
    for i in range(10):
        epochs = [] # array containing epochs for specific M
        alpha = []  # array containing learning rate for specific M
        for j in range(list_M.size):
            if list_M[j] == M_value:
                epochs.append(list_epochs[j])
                alpha.append(list_alpha[j])
        plt.plot(alpha, epochs, color=colors[i], label='M='+str(M_value))
        M_value *= 2

    plt.xlabel("Learning rate")
    plt.ylabel("Epochs")
    plt.title("Epochs (E) compared to learning rate (η) \nfor different sizes of hidden layer (M)")

    plt.legend()
    plt.savefig("compare epochs-learning rate.png")
    print("[Plot has been saved to 'compare epochs-learning rate.png']")
    # plt.show()
