# -*- coding: utf-8 -*-

import operator
from classifier import Classifier
from collections import Counter, defaultdict

import numpy as np


class PoissonNaiveBayes(Classifier):

    """Sparse matrix multinomial Naive Bayes classifier."""

    def __init__(self, feature_count, class_count):
        self._feature_count = feature_count
        self._class_count = class_count
        self._document_count = 0
        self._class_map = {}
        self._p_x = np.zeros((class_count, feature_count), dtype=np.float64)
        self._document_class_count = np.zeros(class_count, dtype=np.int)

    def _frequency(self, x, document_length, theta=1, tau=2000):
        f = (x + theta) / (document_length + theta * self._feature_count)
        return f * tau

    def _index_for_class(self, class_type):
        if class_type in self._class_map:
            return self._class_map[class_type]

        curr_class_count = len(self._class_map)
        self._class_map[class_type] = curr_class_count
        return curr_class_count

    def partial_fit(self, X, y):
        for j in range(len(X)):
            features = X[j]
            class_type = y[j]

            document_length = sum(features)
            class_index = self._index_for_class(class_type)
            for i, x in enumerate(features):
                f_ij = self._frequency(x, document_length)
                self._p_x[class_index][i] += f_ij

            self._document_count += 1
            self._document_class_count[class_index] += 1

    def fit(self, X, y):
        return self.partial_fit(X, y)

    def predict_one(self, x):
        doc_length = sum(x)
        f_vector = list(map(lambda x_i: self._frequency(x_i, doc_length), x))

        max_prob = float("-inf")
        max_class = None

        for class_type, class_index in self._class_map.items():
            p = self._probability_for_class(f_vector, class_index)
            if p > max_prob:
                max_prob = p
                max_class = class_type

        return max_class

    def _probability_for_class(self, f_vector, class_index):
        class_count = self._document_class_count[class_index]
        non_class_count = self._document_count - class_count

        log_prob = 0.00

        for i in range(len(f_vector)):
            f = f_vector[i]

            class_f = self._p_x[class_index][i]
            non_class_f = sum(self._p_x[:, i]) - class_f

            class_mean = class_f / class_count
            non_class_mean = non_class_f / non_class_count

            log_prob += np.log(self._poisson(f, class_mean))
            log_prob -= np.log(self._poisson(f, non_class_mean))

        p = np.e ** log_prob
        p_c = class_count / float(self._document_count)
        p_not_c = 1.0 - p_c

        return p * p_c / (p * p_c + p_not_c)

    def _poisson(self, f, mean):
        return np.e ** (-mean) * mean ** f


class MultinomialNaiveBayes(Classifier):

    """Multinomial Naive Bayes classifier."""

    def __init__(self):
        """Constructs a MultinomialNaiveBayes classifier."""
        self._p_x = {}
        self._p_y = {}
        self._n = 0

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Input matrix.
            y: Labeled output vector.
        """
        self._n = len(y)
        self._p_y = dict(Counter(list(y)))

        _, feature_count = X.shape
        self._p_x = {}
        for i in range(self._n):
            k = y[i]
            if k not in self._p_x:
                self._p_x[k] = [defaultdict(int) for _ in range(feature_count)]

            for j in range(feature_count):
                x_j = self._p_x[k][j]
                x_class = X[i][j]
                if x_class not in x_j:
                    x_j[x_class] = 0
                x_j[x_class] += 1

    def predict_one(self, x):
        """Predicts the output label for a single data element.

        Args:
            x: Input features.

        Returns:
            Labeled output.
        """
        probs = {
            k: self._probability_for_class(x, k)
            for k in self._p_y
        }

        return max(probs.items(), key=operator.itemgetter(1))[0]

    def _probability_for_class(self, x, k):
        """Computes the probability of a an input belonging to a certain class.

        Args:
            x: Input features.
            k: Output class.

        Returns:
            Probability.
        """
        p = self._p_y[k] / self._n
        for i, j in enumerate(x):
            p_xj_y = (self._p_x[k][i][j] + 1) / (self._p_y[k] + 2)
            p *= p_xj_y
        return p
