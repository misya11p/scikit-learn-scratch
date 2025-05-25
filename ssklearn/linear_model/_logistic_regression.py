import numpy as np
from .._base import BaseClassifire


class LogisticRegressionBinary(BaseClassifire):
    def __init__(
        self,
        tol: float = 0.0001,
        max_iter: int = 100,
        fit_intercept: bool = True
    ):
        """logistic regression model."""
        self.tol = tol
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.w = None
        self.iter = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """
        sigmoid function.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: sigmoid function result.
        """
        return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Training model using Newton method.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.
        """
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])
        tol_vec = np.full(X.shape[1], self.tol)
        diff = np.full(X.shape[1], np.inf)
        self.iter = 0
        while np.any(diff > tol_vec) and (self.iter < self.max_iter):
            pred = self._sigmoid(np.dot(X, self.w))
            grad = np.dot(X.T, (pred - y))
            hess = np.dot(X.T * pred * (1 - pred), X)
            hess_inv = np.linalg.pinv(hess)
            w_new = self.w - np.dot(hess_inv, grad)
            diff = w_new - self.w
            self.iter += 1
            self.w = w_new

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability using the model.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Probability prediction results.
        """
        if self.fit_intercept:
            X = np.insert(X, 0, 1, axis=1)
        return self._sigmoid(np.dot(X, self.w))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Binary prediction results.
        """
        return (self.predict_proba(X) > 0.5).astype(int)

