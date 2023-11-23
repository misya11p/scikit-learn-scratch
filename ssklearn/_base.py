from abc import ABCMeta, abstractmethod
import numpy as np
from .metrics import accuracy_score, r2_score


class BaseClassifire(metaclass=ABCMeta):
    """Base class for all classifires"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return accuracy score.

        Args:
            X (np.ndarray):
                Test samples. 2d numpy array of shape
                (n_samples, n_features).
            y (np.ndarray):
                True labels for `X`. 1d numpy array  of shape
                (n_samples,).

        Returns:
            float: Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        return accuracy_score(y, self.predict(X))


class BaseRegressor(metaclass=ABCMeta):
    """Base class for all regressors"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return R^2 score.

        Args:
            X (np.ndarray):
                Test samples. 2d numpy array of shape
                (n_samples, n_features).
            y (np.ndarray):
                True values for `X`. 1d numpy array  of shape
                (n_samples,).

        Returns:
            float: R^2 score of `self.predict(X)` w.r.t. `y`.
        """
        return r2_score(y, self.predict(X))


class BaseCluster(metaclass=ABCMeta):
    """Base class for all clusterers"""

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit training data and predict labels.

        Args:
            X (np.ndarray): 2d numpy array of training data.

        Returns:
            np.ndarray: 1d numpy array of predicted labels.
        """
        self.fit(X)
        return self.predict(X)


class BaseTransformer(metaclass=ABCMeta):
    """Base class for all transformers"""

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit training data and transform it.

        Args:
            X (np.ndarray): 2d numpy array of training data.

        Returns:
            np.ndarray: 2d numpy array of transformed data.
        """
        self.fit(X)
        return self.transform(X)
