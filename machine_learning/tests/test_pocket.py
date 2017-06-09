import sys
sys.path.append("..")

import numpy as np

from pocket import Pocket

if __name__ == '__main__':
    X_train = np.array([[1, 2, 3],
                        [2, 3, 1],
                        [-2, -1, -4]])
    y_train = np.array([1, 1, 0]).reshape(-1, 1)
    X_test = np.array([[-1, -2, -3],
                       [2, 1, 1],
                       [-2, -1, -4]])

    pocket = Pocket(fit_intercept=True, eta0=1.0)
    pocket.fit(X_train, y_train)
    print(pocket.predict(X_test))