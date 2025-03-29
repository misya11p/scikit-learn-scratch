import numpy as np
from .._base import BaseRegressor


class LogisticRegression(BaseRegressor):
    def __init__(self, tol: float = 0.0001, max_iter: int = 100):
        """logistic regression model."""
        self.tol = tol
        self.max_iter = max_iter
        self.w = None
        self.iter = None


    @staticmethod
    def _sigmoid(xw: np.ndarray) -> np.ndarray:
        """
        sigmoid function.

        Args:
            xw (np.ndarray): dot product of X and w.

        Returns:
            np.ndarray: sigmoid function result.
        """
        return 1 / (1 + np.exp(-xw))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Training model using Newton method.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        self.w = np.random.randn(X.shape[1])
        tol_vec = np.full(X.shape[1], self.tol)
        diff = np.full(X.shape[1], np.inf)
        self.iter = 0
        while np.any(diff > tol_vec) and (self.iter < self.max_iter):
            y_hat = self._sigmoid(X @ self.w)
            r = y_hat * (1 - y_hat)
            w_new = self.w - np.linalg.solve(
                (X.T * r) @ X,
                X.T @ (y_hat - y)
            )
            diff = w_new - self.w
            self.iter += 1
            self.w = w_new

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Binary prediction results.
        """
        y_hat_xw = np.dot(X, self.w)
        y_pred = self._sigmoid(y_hat_xw)
        for _ in range(X.shape[0]):
            if y_pred[_] > 0.5:
                y_pred[_] = 1
            elif y_pred[_] < 0.5:
                y_pred[_] = 0
        return y_pred
