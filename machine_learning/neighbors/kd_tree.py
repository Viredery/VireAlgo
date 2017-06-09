# Author: viredery

import numpy as np
import heapq

class KDNode:
    def __init__(self, X_row, y_row, split_index, left, right):
        self.X_row = X_row
        self.y_row = y_row
        self.split_index = split_index
        self.left = left
        self.right = right

class KDTree:
    def __init__(self, X, y):
        self.root = self.construct(X, y)

    def construct(self, X, y):
        n_samples, n_features = X.shape
        if n_samples == 0:
            return None
        feature_index = np.argmax(np.var(X, axis=0))
        mid = int(n_samples / 2)
        index = np.argpartition(X[:, feature_index], mid)
        X, y = X[index], y[index]
        X_row, y_row = X[mid], y[mid]

        left_X, left_y = X[:mid], y[:mid]
        left = self.construct(left_X, left_y)

        right_X, right_y = X[mid + 1:], y[mid + 1:]
        right = self.construct(right_X, right_y)

        return KDNode(X_row, y_row, feature_index, left, right)

    def search(self, X_test_row, n_neighbors):
        self.k_neighbors = []
        self.X_test_row = X_test_row
        self.n_neighbors = n_neighbors
        self._search(self.root)
        return [cur_node.y_row for _, cur_node in self.k_neighbors]

    def _search(self, node):
        another_node_inspection = False
        cur_node = node
        if cur_node is None:
            return
        X_train_row = cur_node.X_row
        distance = np.linalg.norm(self.X_test_row - X_train_row)

        if self.X_test_row[cur_node.split_index] < X_train_row[cur_node.split_index]:
            target_node, another_node = cur_node.left, cur_node.right
        else:
            target_node, another_node = cur_node.right, cur_node.left

        self._search(target_node)

        # backtracking
        if len(self.k_neighbors) < self.n_neighbors:
            neg_distance = -1 * distance
            heapq.heappush(self.k_neighbors, (neg_distance, cur_node))
            another_node_inspection = True
        else:
            top_neg_distance, top_node = heapq.heappop(self.k_neighbors)
            if - 1 * top_neg_distance > distance:
                top_neg_distance, top_node = -1 * distance, cur_node
            top_neg_distance, top_node = heapq.heappushpop(self.k_neighbors, (top_neg_distance, top_node))
            if abs(self.X_test_row[cur_node.split_index] - X_train_row[cur_node.split_index]) < -1 * top_neg_distance:
                another_node_inspection = True
            heapq.heappush(self.k_neighbors, (top_neg_distance, top_node))

        if another_node_inspection:
            self._search(another_node)
