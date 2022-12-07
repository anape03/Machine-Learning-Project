from LoadData import load_data                                   # Part A
from LogisticRegression import resultsLogRegr, resultsLogRegrL2  # Part B 
def main():
    # Part A ---------------------------------

    x_train, x_test, x_val, y_train, y_test, y_val = load_data()
    print("[Data has been loaded.]")
    
    print("-"* 30)

    xdata = [x_train, x_test, x_val]
    ydata = [y_train, y_test, y_val]

    # Part B ---------------------------------

    print("[Logistic Regression...]")
    resultsLogRegr(xdata, ydata)

    print("-"* 30)

    print("[Logistic Regression using L2 regularization...]")
    resultsLogRegrL2(xdata, ydata)

    # Part C ---------------------------------


if __name__ == "__main__":
    main()
