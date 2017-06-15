# Author: viredery

import numpy as np

class LogisticRegression:
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, eta0=1.0, max_iter=100):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.eta0 = eta0
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        self.weights_, self.intercept_ = np.zeros(n_features + 1), np.zeros(1)
        index = 0
        for _ in range(self.max_iter):
            gradiant = np.dot(self._sigmoid(X) - y, X)
            if self.penalty == 'l2':
                self.weights_ = self.weights_ - self.eta0 * (gradiant + 1 / self.C * self.weights_)
            else:
            	self.weights_ = self.weights_ - self.eta0 * gradiant
        if self.fit_intercept:
            self.intercept_ = self.weights_[0]
            self.weights_ = self.weights_[1:]

    def predict(self, X):
        if self.fit_intercept == False:
            n_samples, n_features = X.shape
            X = np.hstack((np.ones((n_samples, 1)), X))
        sigmoid = self._sigmoid(X)
        return np.where(sigmoid > 0.5, 1, 0)

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.weights_) - self.intercept_))