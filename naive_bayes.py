# -*- coding: utf-8 -*-

import operator
from classifier import Classifier
from collections import Counter, defaultdict


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
