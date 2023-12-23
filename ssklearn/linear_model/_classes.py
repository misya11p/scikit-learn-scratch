import numpy as np


class SimpleLinearRegression:
    def fit(self, x, y):
        var_x, cov = np.cov(x, y)[0]
        bar_x = x.mean()
        bar_y = y.mean()
        self.a = cov / var_x
        self.b = bar_y - self.a * bar_x

    def predict(self, x):
        y = self.a * x + self.b
        return y
