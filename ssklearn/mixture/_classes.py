import numpy as np
from scipy.stats import multivariate_normal
from .._base import BaseCluster


class GaussianMixture(BaseCluster):
    def __init__(self, n_components: int, max_iter: int = 100):
        """Gaussian mixture model with EM algorithm for clustering."""
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X: np.ndarray):
        """
        Training model using EM algorithm.

        Args:
            X (np.ndarray): Training data.
        """
        N, d = X.shape
        self._init_params(d)

        for _ in range(self.max_iter):
            gamma = self._posterior_dist(X)
            Nk = gamma.sum(axis=1)

            self.mu = gamma @ X / Nk.reshape(-1, 1)

            dev = np.array([X - self.mu[k] for k in range(self.n_components)])
            dev_gamma = (gamma[..., np.newaxis] * dev).transpose(0, 2, 1)
            self.sigma = dev_gamma @ dev / Nk.reshape(-1, 1, 1)

            self.pi = Nk / N

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster using the model. Return the cluster with the
        highest posterior probability.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Predicted clusters.
        """
        gamma = self._posterior_dist(X)
        c = gamma.argmax(axis=0)
        return c

    def _init_params(self, d: int):
        """
        Initialize parameters.

        - mu: Randomly initialized.
        - sigma: Identity matrix.
        - pi: Uniform distribution.

        Args:
            d (int): Number of features.
        """
        self.mu = np.random.randn(self.n_components, d)
        self.sigma = np.tile(np.eye(d), (self.n_components, 1, 1))
        self.pi = np.ones(self.n_components) / self.n_components

    def _posterior_dist(self, X: np.ndarray) -> np.ndarray:
        """
        Compute posterior distribution.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Posterior distribution.
        """
        joint_probs = []
        for mu, sigma, pi in zip(self.mu, self.sigma, self.pi):
            likelihood = multivariate_normal.pdf(X, mean=mu, cov=sigma)
            joint_prob = pi * likelihood
            joint_probs.append(joint_prob)
        joint_probs = np.array(joint_probs)
        gamma = joint_probs / joint_probs.sum(axis=0)
        return gamma
