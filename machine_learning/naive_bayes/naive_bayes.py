# Author: viredery

import numpy as np

class BernoulliNB:
    """Naive Bayes Classifier for Bernoulli Models
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = 2

        if self.class_prior == None:
            if not self.fit_prior:
                self.class_prior_ = [1 / n_classes] * n_classes
            else:
                self.class_prior_ = (np.bincount(y)[self.classes_] + self.alpha) / (n_samples + self.alpha * n_classes)
        else:
            self.class_prior_ = self.class_prior

        self.conditional_prob_ = {}
        # metrix: [n_classes, n_features]
        for c in self.classes_:
            self.conditional_prob_[c] = {}
            for feature_index in range(n_features):
                feature = X[np.equal(y,c)][:,feature_index]

                binary_probs = (len(feature[feature == 1]) + self.alpha) / (len(feature) + self.alpha * 2)

                self.conditional_prob_[c][feature_index] = binary_probs

        return self

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=1, arr=X)

    def _predict(self, X_row):
        posterior_prob = self.class_prior_.copy()
        for i, c in enumerate(self.classes_):
            for feature_index in range(len(X_row)):
                if X_row[feature_index] == 1:
                    posterior_prob[i] *= self.conditional_prob_[c][feature_index]
                else:
                    posterior_prob[i] *= 1 - self.conditional_prob_[c][feature_index]
        return self.classes_[np.argmax(posterior_prob)]


class MultinomialNB:
    """Naive Bayes Classifier for Multinomial Models

    parameters
    ----------
    alpha : double
        Additive (Laplace/Lidstone) smoothing parameter

    fit_prior : double
        Whether to learn class prior probabilities or not.

    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.class_prior == None:
            if not self.fit_prior:
                self.class_prior_ = [1 / n_classes] * n_classes
            else:
                self.class_prior_ = (np.bincount(y)[self.classes_] + self.alpha) / (n_samples + self.alpha * n_classes)
        else:
            self.class_prior_ = self.class_prior

        self.conditional_prob_ = {}
        for c in self.classes_:
            self.conditional_prob_[c] = {}
            for feature_index in range(n_features):

                feature = X[np.equal(y,c)][:,feature_index]

                values = np.unique(feature)
                value_probs = (np.bincount(feature)[values] + self.alpha) / (len(feature) + self.alpha * len(values))

                self.conditional_prob_[c][feature_index] = dict(zip(values, value_probs))

        return self
    
    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=1, arr=X)

    def _predict(self, X_row):
        posterior_prob = self.class_prior_.copy()
        for i, c in enumerate(self.classes_):
            for feature_index in range(len(X_row)):
                if X_row[feature_index] in self.conditional_prob_[c][feature_index]:
                    posterior_prob[i] *= self.conditional_prob_[c][feature_index][X_row[feature_index]]
                else:
                    posterior_prob[i] *= 1 # If this value is not in train data, we cannot smooth the probability.
        return self.classes_[np.argmax(posterior_prob)]

class GaussianNB:
    """Naive Bayes Classifier for Gaussian Models

    alpha : double
        Additive (Laplace/Lidstone) smoothing parameter

    fit_prior : double
        Whether to learn class prior probabilities or not.

    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not adjusted according to the data.

    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.class_prior == None:
            if not self.fit_prior:
                self.class_prior_ = [1 / n_classes] * n_classes
            else:
                self.class_prior_ = (np.bincount(y)[self.classes_] + self.alpha) / (n_samples + self.alpha * n_classes)
        else:
            self.class_prior_ = self.class_prior

        self.conditional_prob_ = {}
        for c in self.classes_:
            self.conditional_prob_[c] = {}
            for feature_index in range(n_features):
                feature = X[np.equal(y,c)][:,feature_index]

                mu = np.mean(feature)
                sigma = np.std(feature)

                self.conditional_prob_[c][feature_index] = (mu, sigma)

        return self
    
    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=1, arr=X)

    def _predict(self, X_row):
        posterior_prob = self.class_prior_.copy()
        for i, c in enumerate(self.classes_):
            for feature_index in range(len(X_row)):
                mu, sigma = self.conditional_prob_[c][feature_index]
                posterior_prob[i] *= (1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (X_row[feature_index] - mu) ** 2 / (2 * sigma ** 2)))
        return self.classes_[np.argmax(posterior_prob)]