# Author: viredery

import sys
sys.path.append("..")

import numpy as np

from perceptron import Perceptron

if __name__ == '__main__':
    X_train, y_train = np.array([[1, 2, 3], [2, 3, 1], [-2, -1, -4]]), np.array([1, 1, 0])
    X_test = np.array([[-1, -2, -3], [2, 1, 1], [-2, -1, -4]])

    perceptron = Perceptron(fit_intercept=True, eta0=1)
    perceptron.fit(X_train, y_train)
    print(perceptron.predict(X_test))