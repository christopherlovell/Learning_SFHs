import numpy as np

def mse(true, pred):
    """
    Mean squared error
    """
    return np.sum(pow(true - pred, 2)) / len(true)


def mae(true, pred):
    """
    Mean absolute error
    """
    return np.sum(np.abs(true - pred)) / len(true)


def absolute_error(true, pred):
    """
    Absolute error
    """
    return np.sum(np.abs(true - pred))


def _SMAPE(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    
    We use the version ignoring the factor of 2 in the denominator, leading
    to a percentage error between 0 and 100 (where 0 is a perfect prediction).
    
    SMAPE = 1 / n * Sum[ |Yp - Yt| / (|Yt| + |Yp|)]
    """

    return np.sum(np.abs(y_pred - y_true)) / np.sum(y_pred + y_true)


def _R_squared(y_true, y_pred):
    """
    R^2 error
    """

    return 1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2)


from keras import backend as K

def _SMAPE_tf(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    
    We use the version ignoring the factor of 2 in the denominator, leading
    to a percentage error between 0 and 100 (where 0 is a perfect prediction).
    
    SMAPE = 1 / n * Sum[ |Yp - Yt| / (|Yt| + |Yp|)]
    """
    return K.sum(K.abs(y_pred - y_true)) / K.sum(y_pred + y_true)


def _R_squared_tf(y_true, y_pred):
    """
    negative R^2 error
    """

    return -1 * (1 - K.sum((y_pred - y_true)**2) / K.sum((y_true - K.mean(y_true))**2))

    