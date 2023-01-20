import numpy as np

class MLP():
    def __init__(self, input_size, hidden_size, output_size, alpha = 0.001):
        """
        Initialise MLP model

        Args:
            input_size  : number of neurons in first layer
            hidden_size : number of neurons in hidden layer
            output_size : number of neurons in output layer
            alpha       : learning rate
        """
        # Number of neurons for each layer
        self.input_dim  = input_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size

        # Learning rate 
        self.alpha = alpha 

        # Initialise weights
        self.w1 = np.random.randn(self.input_dim,  self.hidden_dim) * 0.001
        self.w2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.001

        # Initialise bias
        self.b1 = np.ones((1, self.hidden_dim))
        self.b2 = np.ones((1, self.output_dim))

    def forward_prop(self, X):
        """
        Forward propagation

        Args:
            X : numpy array
        Return:
            numpy array
        """
        # First layer
        self.y1 = np.dot(X, self.w1) + self.b1
        self.z1 = sigmoid(self.y1)

        # Second layer
        self.y2 = np.dot(self.z1, self.w2) + self.b2
        self.z2 = sigmoid(self.y2)

        return self.z2
    
    def back_prop(self, X, y):
        """
        Backward propagation, and gradient descent

        Args:
            X : numpy array
            y : numpy array
        """
        e1 = np.subtract(self.z2, y)
        self.dW2 = e1 * sigmoid_deriv(self.z2)
    
        e2 = np.dot(e1, self.w2.T)
        self.dW1 = e2 * sigmoid_deriv(self.z1)

        # Gradient descent 
        # Update weights
        w1_update = np.dot(      X.T, self.dW1) / y.shape[0]
        w2_update = np.dot(self.z1.T, self.dW2) / y.shape[0]
        
        self.w1 -= w1_update * self.alpha
        self.w2 -= w2_update * self.alpha

        # Update bias
        b1_update = np.sum(e2, axis=0, keepdims=True) / X.shape[1]
        b2_update = np.sum(e1, axis=0, keepdims=True) / X.shape[1]

        self.b1 -= b1_update * self.alpha
        self.b2 -= b2_update * self.alpha


def sigmoid(x):
    '''
    Applies sigmoid function to every item in array x
    ( squashes the output of the linear function into the
    interval (0, 1) and interpret that value as a probability )

    Args:
        x : numpy array
    Return:
        numpy array
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(z):
    """
    Sigmoid first order derivative

    Args:
        z : numpy array
    Return:
        numpy array
    """
    return z * (1 - z)

def cost_grad_sigmoid(W, X, t, lamda):
    E = 0
    N, D = X.shape
    K = t.shape[1]

    y = sigmoid(np.dot(X, W))

    for n in range(N):
        for k in range(K):
            E += t[n][k] * np.log(y[n][k])
    E -= lamda * np.sum(np.square(W)) / 2

    gradEw = np.dot((t - y).T, X) - lamda * W.T
    return E, gradEw

def gradcheck_sigmoid(Winit, X, t, lamda):
    W = np.random.rand(*Winit.shape)
    epsilon = 1e-6

    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    Ew, gradEw = cost_grad_sigmoid(W, x_sample, t_sample, lamda)
    print("gradEw shape: ", gradEw.shape)

    numericalGrad = np.zeros(gradEw.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[d, k] += epsilon
            e_plus, _ = cost_grad_sigmoid(w_tmp, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(W)
            w_tmp[d, k] -= epsilon
            e_minus, _ = cost_grad_sigmoid(w_tmp, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    return gradEw, numericalGrad

def crossEntropyLoss(y_hat, Y):
    """
    Calculate Cross-entropy cost

    Args:
        y_hat : predicted values
        Y     : actual values
    """
    t1 = np.dot(Y.T, np.log(y_hat))
    t2 = np.dot((1-Y).T, np.log(1 - y_hat))

    prob = t1 + t2
    cost = - prob / Y.shape[0]

    return float(np.squeeze(cost)) 

def gradient_checking(X, y, input_size=784, hidden_size=100, output_size=1, alpha=0.001):
    """
    Args:
        input_size  : number of neurons in first layer
        hidden_size : number of neurons in hidden layer
        output_size : number of neurons in output layer
        alpha       : learning rate
    """
    model = MLP(input_size, hidden_size, output_size, alpha)

    model.forward_prop(X)
    model.back_prop(X, y)
    gradEw, numericalGrad = gradcheck_sigmoid(model.w1, X, y, 0.1)
    print("The difference estimate for gradient of w1 is : ", np.max(np.abs(gradEw - numericalGrad)))
    gradEw, numericalGrad = gradcheck_sigmoid(model.w2, sigmoid(np.dot(X, model.w1) + model.b1), y, 0.1)
    print("The difference estimate for gradient of w2 is : ", np.max(np.abs(gradEw - numericalGrad)))

def trainModel(xdata, ydata, input_size=784, hidden_size=100, output_size=1, tolerance=5, round_to=4, alpha=0.001):
    """
    Train model, using early stopping.
    After every epoch, the model's mean cost is calculated on validation set.
    Stop training if there hasn't been a (considerable) descrease for 5 consecutive epochs, or there's an increase.
    
    Args:
        input_size  : number of neurons in first layer
        hidden_size : number of neurons in hidden layer
        output_size : number of neurons in output layer
        tolerance   : exit if after this number of epochs there hasn't been a reduction in cost, or there has been an increase
        round       : number to round cost to (for early stopping)
        alpha       : learning rate
    """
    x_train, x_test, x_val = xdata
    y_train, y_test, y_val = ydata

    model = MLP(input_size, hidden_size, output_size, alpha)

    cost_list = []  # list with cost calculated in each epoch
    epoch = 0
    prev_cost = 0   # cost of previous epoch
    counter = 0     # counts the number of epochs that (consecutively) didn't have a good enough reduction in cost, or had an increse
    while(True):
        # Forward Propagation
        model.forward_prop(x_train)

        # Calculate cost --------------
        # First layer
        y1 = np.dot(x_val, model.w1) + model.b1
        z1 = sigmoid(y1)

        # Second layer
        y2 = np.dot(z1,    model.w2) + model.b2
        y_hat = sigmoid(y2)

        cost = crossEntropyLoss(y_hat, y_val)
        cost_list.append(cost)
        curr_cost = round(np.mean(cost_list), round_to) # current mean cost
        # -----------------------------

        # Check exit
        if prev_cost <= curr_cost:
            counter += 1
            if counter == tolerance:
                break
        else:
            counter = 0
        
        if epoch % 500 == 0:
           print (f"Calculated cost (epoch {epoch}): {curr_cost}")
        
        # Backward Propagation
        model.back_prop(x_train, y_train)

        prev_cost = curr_cost
        epoch += 1

    print(f"Stopped at epoch '{epoch}' with final cost: {curr_cost}")
    return epoch, cost, model

def testParameters(xdata, ydata, input_size=784, output_size=1, tolerance=5, round_to=4):
    """
    Train model using trainModel() for different values of 
    learning rate, and number of neurons in hidden layer.

    Args:
        input_size  : number of neurons in first layer
        output_size : number of neurons in output layer
        tolerance   : exit if after this number of epochs there hasn't been a reduction in cost, or there has been an increase
        round       : number to round cost to (for early stopping)
    """
    best = { # Data used for best calculated model
        "epoch" : -1,            # epoch
        "cost"  : float('inf'),  # calculated cost
        "alpha" : -1,            # learning rate
        "M"     : -1,            # number of neurons in hidden layer
        "model" : None,
    }
    data = { # Data saved from all models
        "epochs" : [],
        "alpha"  : [],
        "M"      : [],
    }

    min_alpha, max_alpha = 1e-5, 0.5
    for a in np.linspace(min_alpha, max_alpha, 10): # 10 different values for learning rate (a), in [1e-5, 0.5]
        M = 2
        for i in range(10): # 10 different values for neurons in hidden layer (M)
            print("-"*20)
            print(f"[Learning rate: {a}]")
            print(f"[Neurons in hidden layer: {M}]")
            epoch, cost, model = trainModel(xdata, ydata, input_size=input_size, hidden_size=M, output_size=output_size, tolerance=tolerance, alpha=a, round_to=round_to) 
            data["epochs"].append(epoch)
            data["alpha"].append(a)
            data["M"].append(M)
            if cost < best["cost"]:
                best["cost"] = cost
                best["epoch"] = epoch
                best["alpha"] = a
                best["M"] = M
                best["model"] = model
            M *= 2

    print("-"*20)
    print(f"Best calculated model with cost = {round(best['cost'], 4)} and the following parameters:")
    print(f"n = {best['model'].alpha} \t(learning rate)")
    print(f"M = {best['M']} \t(hidden layer)")
    print(f"E = {best['epoch']} \t(epochs)")
    
    return best, data

def testAccuracy(model, x_test, y_test):
    """
    Test accuracy of model on test set.
    """
    y_prediction = model.forward_prop(x_test)
    predictions = np.select([y_prediction < 0.5, y_prediction >= 0.5], [0, 1]) # predicted 0 or 1
    print('Accuracy of test set', round(float(np.squeeze(sum(y_test == predictions) / len(y_test) * 100)), 8))
