# Author: viredery

import sys
sys.path.append("..")

import numpy as np

from logistic import LogisticRegression

if __name__ == '__main__':
    X_train, y_train = np.array([[1, 2, 3], [2, 3, 1], [-2, -1, -4]]), np.array([1, 1, 0])
    X_test = np.array([[-1, -2, -3], [2, 1, 1], [-2, -1, -4]])

    logistic_regression = LogisticRegression(fit_intercept=True, eta0=1.0)
    logistic_regression.fit(X_train, y_train)
    print("predict:")
    print(logistic_regression.predict(X_test))