# Author: viredery

import sys
sys.path.append("..")

import numpy as np

from kd_tree import KDTree

if __name__ == '__main__':
    X_train, y_train = np.array([[1, 2, 3], [2, 3, 1], [-2, -1, -4]]), np.array([1, 1, 0])
    X_test = np.array([[-1, -2, -3], [2, 1, 1], [-2, -1, -4]])

    kd_tree = KDTree(X_train, y_train)
    print(np.array(kd_tree.search(np.array([2, 1, 1]), 3)))
