# Author: viredery

import numpy as np

class Pocket:
    """Pocket Binary Classifier

    parameters
    ----------
    fit_intercept : bool
        Whether the intercept should be estimated or not.

    eta0 : double
        Constant by which the updates are multiplied.

    early_stopping_iteration : int
        Number of updates.

    attributes
    ----------
    coef_ : array, shape = [n_features]
        Weights assigned to the features.

    intercept_ :  array, shape = [1]
        Constant in decision function.

    """
    
    def __init__(self, fit_intercept=True, eta0=1.0, early_stopping_iteration=10):
        self.fit_intercept = fit_intercept
        self.eta0 = eta0
        self.early_stopping_iteration = early_stopping_iteration

    def fit(self, X, y):
        n_samples, n_features = X.shape
        weights, intercept = np.zeros(n_features), np.zeros(1)
        weights_pocket, intercept_pocket = weights, intercept
        loss_pocket = self._calc_zero_one_loss(X, y, weights_pocket, intercept_pocket)
        index = 0
        for _ in range(self.early_stopping_iteration):
            if loss_pocket == 0:
                break
            while y[index] * (np.dot(X[index], weights) + intercept) > 0.0:
                index = (index + 1) % n_samples
            weights += self.eta0 * y[index] * X[index]
            if self.fit_intercept:
                intercept += self.eta0 * y[index]
            loss = self._calc_zero_one_loss(X, y, weights, intercept)
            if loss < loss_pocket:
                weights_pocket, intercept_pocket, loss_pocket = weights, intercept, loss

        self.coef_ = weights_pocket
        self.intercept_ = intercept_pocket

        return self

    def predict(self, X):
        return np.where(np.dot(X, self.coef_) + self.intercept_ > 0, 1, 0)

    def _calc_zero_one_loss(self, X, y, weights, intercept):
        y_pred = np.where(np.dot(X, weights) + intercept > 0, 1, 0)
        return np.not_equal(y, y_pred).sum()
