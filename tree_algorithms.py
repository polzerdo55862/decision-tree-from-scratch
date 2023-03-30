from sklearn.tree import DecisionTreeRegressor
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from collections import Counter

class Node():
    def __init__(
                    self, 
                    feature=None, 
                    threshold=None, 
                    left=None, 
                    right=None, 
                    value=None
                ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # is it a leave node?

    def is_leaf_node(self):
        return self.value is not None

class RegressionTree():
    def __init__(
                    self, 
                    min_samples_split=2, 
                    max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # check the stopping criteria
        n_samples, n_feats = X.shape
    
        if (depth>=self.max_depth or n_samples<self.min_samples_split):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)

        # find the best split
        best_feature_ixd, best_threshold = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature_ixd], best_threshold)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature_ixd, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        y_mean = np.mean(y)
        residuals_y = (y - y_mean)**2
        y_mse = np.mean(residuals_y)
        
        best_feature_ixd, best_threshold = None, None
        lowest_mse = y_mse

        for feat_idx in feat_idxs:
            # define possible thresholds for the split
            X_column = X[:, feat_idx]
            thresholds = np.convolve(np.sort(X_column), np.ones(2)/2, mode='valid')

            for threshold in thresholds:
                # getting the left and right nodes
                left_idxs, right_idxs = self._split(X_column, threshold)

                # calculate the weighted avg. mse of children
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                mse_l = self._squared_error(y[left_idxs]) 
                mse_r = self._squared_error(y[right_idxs])
                child_mse = (n_l/n) * mse_l + (n_r/n) * mse_r

                if lowest_mse > child_mse:
                    lowest_mse = child_mse
                    best_feature_ixd = feat_idx
                    best_threshold = threshold

        return best_feature_ixd, best_threshold

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _squared_error(self, y):
        # calculate the mean value for all observations
        y_mean = np.mean(y)

        # calculate the residuals to y_mean
        mean_squared_error = np.mean((y - y_mean)**2)

        return mean_squared_error

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)





 