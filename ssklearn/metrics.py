import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy classification score.

    Args:
        y_true (np.ndarray): 1d numpy array of correct class labels.
        y_pred (np.ndarray): 1d numpy array of predicted class labels.

    Returns:
        float: Accuracy score.
    """
    return np.mean(y_true == y_pred)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R2 (coefficient of determination) regression score function.

    Args:
        y_true (np.ndarray): 1d numpy array of correct target values.
        y_pred (np.ndarray): 1d numpy array of predicted target values.

    Returns:
        float: R2 score.
    """
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - sse/sst
