import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ----------------------------------------
# Part A ---------------------------------

def load_data():
    """
    Load data from tensorflow, and edit.
    * Rescale images from [0,255] to [0.0,1.0]
    * Filter classes 5 and 6 from dataset
    * Split Train set to Train/Validation (80/20)
    * Tranform images to vectors of 784 elements

    Return:
        Train, Test and Validation data
    """
    print("[Loading Data...]")
    # Load raw data --------------------------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("[Editing Data...]")
    # Rescale the images from [0,255] to [0.0,1.0] range -------
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    # Filter classes -------------------------------------------
    x_train, y_train = filter_classes(x_train, y_train)
    x_test, y_test = filter_classes(x_test, y_test)

    # Split train set (train/validation, 80/20) ----------------
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

    # Tranform images to vectors of 784 elements ---------------
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_val = x_val.reshape(x_val.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)
    
    return x_train, x_test, x_val, y_train, y_test, y_val


def filter_classes(x, y, classes=[5,6]):
    """
    Filter the dataset to keep just the 5s and 6s,
    (by default, can be changed) 
    and remove the other classes. 
    Convert the label, y, to boolean: 
        0 for 5 and 1 for 6. 
    """
    keep = (y == classes[1]) | (y == classes[0])
    x, y = x[keep], y[keep]
    y = y == classes[1]
    return x,y