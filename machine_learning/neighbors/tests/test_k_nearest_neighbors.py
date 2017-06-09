# Author: viredery

import sys
sys.path.append("..")

import numpy as np

from k_nearest_neighbors import KNeighborsClassifier

if __name__ == '__main__':
    X_train, y_train = np.array([[1, 2, 3], [2, 3, 1], [-2, -1, -4]]), np.array([1, 1, 0])
    X_test = np.array([[-1, -2, -3], [2, 1, 1], [-2, -1, -4]])

    k_neighbors_classifier = KNeighborsClassifier(n_neighbors=2, algorithm='kd_tree', metric='euclidean')
    k_neighbors_classifier.fit(X_train, y_train)
    print(k_neighbors_classifier.predict(X_test))

    k_neighbors_classifier = KNeighborsClassifier(n_neighbors=2, algorithm='brute', metric='euclidean')
    k_neighbors_classifier.fit(X_train, y_train)
    print(k_neighbors_classifier.predict(X_test))