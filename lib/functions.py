import numpy as np
from abc import ABC
from scipy.special import softmax


_sigma = lambda x : 1 / (1 + np.exp(-x))


class Function(ABC):
    def loss(self, x):
        pass
    
    def gradient(self, x):
        pass

    def getDimension(self):
        pass

    def getInitialX(self):
        pass


class LogisticRegression(Function):
    def __init__(self, X, y, regularization=None):
        self._N = X.shape[0]
        self._dim = X.shape[1] + 1
        self._X = (np.insert(X.T, 0, np.ones(self._N)).reshape(self._dim, self._N)).T
        self._y = y
        self.regularization = self._none_regularization
        self._lambda = 0.05
        if regularization == 'l2':
            self.regularization = self._l2_regularization

    def _none_regularization(self, _):
        return 0

    def _l2_regularization(self, theta):
        return 2 * self._lambda * theta

    def loss(self, theta):
        loss = 0
        for i in range(self._N):
            loss += (
                self._y[i] * np.log(_sigma(theta @ self._X[i].T))
                + (1 - self._y[i]) * np.log(1 - _sigma(theta @ self._X[i].T))
            )
        return loss

    def gradient(self, theta):
        res = np.zeros(self._dim)
        for i in range(self._N):
            res += (self._y[i] - _sigma(theta @ self._X[i].T)) * self._X[i]
        return res - self.regularization(theta)

    def getDimension(self):
        return self._dim

    def getInitialX(self):
        return np.zeros(self._dim)
