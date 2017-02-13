# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from classifier import Classifier
from sklearn.preprocessing import label_binarize


def preprocess_output(y):
    """Preprocesses output vector.

    Args:
        y: Labeled output vector.

    Returns:
        (Array of class mappings, class counts).
    """
    class_counter = Counter(y)
    classes = np.array(list(class_counter.keys()))
    class_count = np.array(list(class_counter.values()),
                           dtype=np.float)
    return classes, class_count


def compute_log_prob(x, axis=None):
    """Computes the log probability.

    Args:
        x: Input count matrix.
        axis: Axis to sum over.

    Returns:
        Log probability.
    """
    return np.log(x) - np.log(x.sum(axis=axis).reshape(-1, 1))


class MultinomialNaiveBayes(Classifier):

    """Multinomial Naive Bayes classifier."""

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Input matrix.
            y: Labeled output vector.
        """
        # Compute priors.
        self.classes_, self.class_count_ = preprocess_output(y)
        self.class_log_prior_ = compute_log_prob(self.class_count_)

        # Compute conditional probabilities.
        Y = label_binarize(y, self.classes_)
        self.feature_count_ = Y.T * X + 1
        self.feature_log_prob_ = compute_log_prob(self.feature_count_, axis=1)

    def predict(self, X):
        """Predicts the output labels.

        Args:
            X: Input matrix.

        Returns:
            Labeled output vector.
        """
        p = X * self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(p, axis=1)]
