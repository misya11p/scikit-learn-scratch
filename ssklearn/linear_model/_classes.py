import numpy as np


class SimpleLinearRegression:
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
