from abc import ABCMeta, abstractmethod
import numpy as np
from .metrics import accuracy_score, r2_score


class BaseClassifire(meta=ABCMeta):
    """Base class for all classifires"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Return accuracy score.

        Args:
            y_true (np.ndarray): 1d numpy array of correct class labels.
            y_pred (np.ndarray): 1d numpy array of predicted class labels.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred)


class BaseRegression(meta=ABCMeta):
    """Base class for all regressors"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Return R2 score.

        Args:
            y_true (np.ndarray): 1d numpy array of correct target values.
            y_pred (np.ndarray): 1d numpy array of predicted target values.

        Returns:
            float: R2 score.
        """
        return r2_score(y_true, y_pred)


class BaseCluster(meta=ABCMeta):
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


class BaseTransformer(meta=ABCMeta):
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
