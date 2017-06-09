# Author: viredery

import numpy as np

from kd_tree import KDTree

class KNeighborsClassifier:
    """K-Nearest Neighbors Classifier
    
    parameters
    ----------
    n_neighbors : bool
        Number of neighbors to use.

    algorithm : {'kd_tree', 'brute'}
        Algorithm used to compute the nearest neighbors.

    metric : {'euclidean', 'manhattan', 'chebyshev'}
        The distance metric.
 
    """

    def __init__(self, n_neighbors=5, algorithm='brute', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric

    def fit(self, X, y):
        self._fit_X = X
        self._fit_y = y

    def predict(self, X):
        if self.algorithm == 'brute':
            return np.apply_along_axis(self._get_prediction_brute_search, axis=1, arr=X)
        if self.algorithm == 'kd_tree':
            self.kd_tree = KDTree(self._fit_X, self._fit_y)
            return np.apply_along_axis(self._get_prediction_kdtree_search, axis=1, arr=X)
        
    def _get_prediction_brute_search(self, X_row):
        diff = self._fit_X - np.tile(X_row, (self._fit_X.shape[0], 1))
        distances = None

        if self.metric == 'manhattan':
            distances = np.sum(np.abs(diff), axis=1)
        elif self.metric == 'chebyshev':
            distances = np.max(np.abs(diff), axis=1)
        else:
            distances = np.linalg.norm(diff, axis=1)

        k_neighbors = self._fit_y[np.argsort(distances)[:self.n_neighbors]]
        return np.argmax(np.bincount(k_neighbors))

    def _get_prediction_kdtree_search(self, X_row):
    	self.metric = 'euclidean' # the euclidean distance metric can be accepted by kd-Tree search
        return np.argmax(np.bincount(self.kd_tree.search(X_row, self.n_neighbors)))
