from LoadData import load_data                                                 # Part A
from LogisticRegression import resultsLogRegr, resultsLogRegrL2                # Part B 
from MLP import gradient_checking, trainModel, testParameters, testAccuracy    # Part C
from plots import plotAllModels

import json
import numpy as np
from csv import writer

def main():
    # Part A ---------------------------------

    x_train, x_test, x_val, y_train, y_test, y_val = load_data()
    print("[Data has been loaded.]")
    
    print("-"* 30)

    xdata = [x_train, x_test, x_val]
    ydata = [y_train, y_test, y_val]

    # Part B ---------------------------------

    print("[Logistic Regression...]")
    resultsLogRegr(xdata, ydata, round_dec=4)

    print("-"* 30)

    print("[Logistic Regression using L2 regularization...]")
    resultsLogRegrL2(xdata, ydata, round_dec=4)

    # Part C ---------------------------------

    print("-"* 30)

    for i in range(3):
        ydata[i] = ydata[i].reshape(-1,1)

    print("[MLP...]")
    
    # print("[Gradient checking...]")
    # gradient_checking(xdata[0], ydata[0]) # only run once, for testing

    print("-"* 30)

    print("[Training model...]")
    trainModel(xdata, ydata, hidden_size=100, alpha=0.1, round_to=4)

    print("-"* 30)

    print("[Training models for different set of parameters (learning rate, neurons in hidden layer)...]")
    best_model, all_data = testParameters(xdata, ydata, round_to=4)

    # Save data of best model to file
    data = json.dumps(best_model, default=lambda o: '<not serializable>')
    with open("best-model.json","w") as f:
        f.write(data)

    # Save all data to file
    arr = np.asarray([ all_data["epochs"], all_data["alpha"], all_data["M"]])
    with open('all_data.csv', 'w', newline='') as file:
        mywriter = writer(file, delimiter=',')
        mywriter.writerows(arr)

    # Create plot diagram for all model data generated
    plotAllModels()

    print("-"* 30)

    # Test Accuracy of best calculated model
    testAccuracy(best_model["model"], xdata[1], ydata[1])


if __name__ == "__main__":
    main()
