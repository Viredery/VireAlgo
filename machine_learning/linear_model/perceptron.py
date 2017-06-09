# Author: viredery

import numpy as np

class Perceptron:
    """Perceptron Binary Classifier

    parameters
    ----------
    fit_intercept : bool
        Whether the intercept should be estimated or not.

    eta0 : double
        Constant by which the updates are multiplied.

    attributes
    ----------
    coef_ : array, shape = [1, n_features]
        Weights assigned to the features.

    intercept_ :  array, shape = [1]
        Constant in decision function.

    """
    
    def __init__(self, fit_intercept=True, eta0=1.0):
        self.fit_intercept = fit_intercept
        self.eta0 = eta0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights, intercept = np.zeros(n_features), np.zeros(1)
        index, num_corrent = 0, 0
        while True:
            if y[index] * (np.dot(X[index], weights) + intercept) <= 0.0:
                weights += self.eta0 * y[index] * X[index]
                if self.fit_intercept:
                    intercept += self.eta0 * y[index]
                num_corrent = 0
            index, num_corrent = (index + 1) % n_samples, num_corrent + 1
            if num_corrent == n_samples:
                break
        
        self.coef_ = weights
        self.intercept_ = intercept
        
        return self

    def predict(self, X):
        return np.where(np.dot(X, self.coef_) + self.intercept_ > 0, 1, 0)
