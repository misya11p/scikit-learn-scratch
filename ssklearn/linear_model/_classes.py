import numpy as np
from .._base import BaseRegressor


class SimpleLinearRegression(BaseRegressor):
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Training model.

        Args:
            x (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        var_x, cov = np.cov(x, y)[0]
        bar_x = x.mean()
        bar_y = y.mean()
        self.a = cov / var_x
        self.b = bar_y - self.a * bar_x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the model.

        Args:
            x (np.ndarray): Samples.

        Returns:
            np.ndarray: Predicted values.
        """
        y = self.a * x + self.b
        return y


class LinearRegression(BaseRegressor):
    def __init__(self):
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Training model.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Predicted values.
        """
        X = np.insert(X, 0, 1, axis=1)
        return X @ self.w
