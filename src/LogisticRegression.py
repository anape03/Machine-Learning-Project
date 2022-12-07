import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Part Β ---------------------------------

class LogisticRegression:
    def __init__(self, iter=100, _lambda=0.0, alpha=0.01):
        self.iter = iter    # number of iterations
        self._lambda = _lambda
        self.alpha = alpha
        self.J_train = []
        self.J_test = []

    def reguralization(self, theta):
        return (self._lambda / (2.0)) * np.sum(np.square(theta))

    def sigmoid(self, x):
        '''
        Applies sigmoid function to every item in array x
        ( squashes the output of the linear function into the
        interval (0, 1) and interpret that value as a probability )

                   1
        S(x) = __________
                     (-x)
                1 + e

        Args:
            x: numpy array
        Return:
            numpy array
        '''
        return 1 / (1 + np.exp(-x))

    def hypothesis(self, X, theta):
        """
        Hypothesis
                          1
        h(x) = __________________________
                     -(SUM[i=0,n] (bi*xi))
                1 + e
        
        Args:
            X: numpy array
            theta: numpy array
        Return:
            numpy array
        """
        return self.sigmoid(np.dot(X, theta))
    
    def ComputeCostGrad(self, X, y, theta):
        h = self.hypothesis(X, theta) 

        # calc cost function
        #       SUM[n=1,m]( yi * ln(h(xi)) + (1-yi) * ln(1-h(xi)) ) - lambda/2 * ||w||**2
        cur_j = (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) - self.reguralization(theta)

        # calculate gradient
        #    SUM[i=1,m]( (h(xi)-yi) * xij ) - lambda * w
        grad = np.mean((y-h) * X.T, axis=1) - self._lambda * theta
        
        return cur_j, grad

    def fit(self, X, y, X_val, y_val):

        theta = np.zeros(X.shape[1])
        
        for i in range(self.iter):    

            train_error, train_grad = self.ComputeCostGrad(X, y, theta)
            test_error, _ = self.ComputeCostGrad(X_val, y_val, theta)
            
            # update parameters by subtracting gradient values
            theta += self.alpha * train_grad

            # store current cost
            self.J_train.append( train_error )
            self.J_test.append( test_error )
            
        return theta

    def predict(self, theta, X):
        """
        Predict whether the label is 0 or 1 using learned logistic regression

        computes the predictions for X using a threshold at 0.5 (i.e., if sigmoid( X * theta ) >= 0.5, predict 1)
        """
        m = X.shape[0]
        
        p = np.zeros((m,1))
        p = self.sigmoid(np.dot(X,theta))
        prob = p
        p = p > 0.5 - 1e-6

        return p, prob


def accuracy(p, y, decimals = None):
    """
    Calculates accuracy

    Args:
        p: numpy array
        y: numpy array
        decimals: integer (if specified, returns result rounded to said decimal)
    Return:
        float
    """
    accuracy = np.mean(p.astype('int') == y)*100
    return accuracy if decimals==None else round(accuracy, decimals)

def accuracyResults(x_train, y_train, x_test, y_test, i=100, _lambda=0.0): 
    """
    Create accuracy results for Logistic Regression.
    
    Args:
        x_train : numpy array
        y_train : numpy array
        x_test  : numpy array
        y_test  : numpy array
        i       : number of iterations
        _lambda : lambda value for model
    Return:
        model(class: LogisticRegression), theta
    """

    model = LogisticRegression(i, _lambda)
    theta = model.fit(x_train, y_train, x_test, y_test)

    return model, theta

def printAccuracy(xdata, ydata, model, theta, iter, _lambda=0.0):
    x_train, x_test, x_val = [data for data in xdata]
    y_train, y_test, y_val = [data for data in ydata]

    p_train, prob_train = model.predict(theta, x_train)
    p_test, prob_test = model.predict(theta, x_test)
    print("-"*10)
    print("For",iter,"iterations ( lambda = ",_lambda,"):")
    print("Accuracy of training set:", accuracy(p_train.astype('int'), y_train, 2))
    print("Accuracy of testing set:", accuracy(p_test.astype('int'), y_test, 2))

def resultsLogRegr(xdata, ydata):
    """
    Creates results for Logistic Regression with no Regularization
    """
    x_train, x_test, x_val = [data for data in xdata]
    y_train, y_test, y_val = [data for data in ydata]
    
    for i in [10, 100, 1000, 10000]:
        model, theta = accuracyResults(x_train, y_train, x_test, y_test, i)
        printAccuracy(xdata, ydata, model, theta, i)

def resultsLogRegrL2(xdata, ydata):
    """
    Creates results for Logistic Regression using L2 Regularization
    """
    x_train, x_test, x_val = [data for data in xdata]
    y_train, y_test, y_val = [data for data in ydata]

    # Test results for different lambda values (on validation set) -------
    print("[Testing accuracy on validation set for different lambda values...]")

    min_l, max_l = 1e-4, 10
    lambda_accuracy = {}

    for _lambda in np.linspace(min_l, max_l, 100): # 100 different lambda values, from 1e-4 to 10
        model, theta = accuracyResults(x_train, y_train, x_val, y_val, 100, _lambda) 
        p_val, prob_val = model.predict(theta, x_val)
        lambda_accuracy[model._lambda] = accuracy(p_val.astype('int'), y_val)
        print(f"Lambda value (accuracy {lambda_accuracy[_lambda]}%): {_lambda}") # TODO: remove

    best_lambda = max(lambda_accuracy, key=lambda_accuracy.get)
    print(f"Best lambda value (accuracy {lambda_accuracy[best_lambda]}%): {best_lambda}")

    # Test accuracy (on test set) using best lambda ----------------------
    print("[Testing accuracy on test set for 'best' lambda value...]")

    model, theta = accuracyResults(x_train, y_train, x_test, y_test, 100, best_lambda)
    p_test, prob_test = model.predict(theta, x_test)
    lambda_accuracy[model._lambda] = accuracy(p_test.astype('int'), y_test, 2)

    print(f"Lambda value (accuracy {lambda_accuracy[best_lambda]}%): {best_lambda}")

