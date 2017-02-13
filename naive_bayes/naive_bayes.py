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


class PoissonNaiveBayes(Classifier):

    """Poisson Naive Bayes classifier."""

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Input matrix.
            y: Labeled output vector.
        """
        # Compute priors.
        self.classes_, self.class_count_ = preprocess_output(y)
        self.class_log_prior_ = compute_log_prob(self.class_count_)

        # Compute means of P(f|c).
        Y = label_binarize(y, self.classes_)
        smoothed_f = self._compute_smoothed_frequency(X)
        self.lambda_ = np.dot(Y.T, smoothed_f)
        self.lambda_ /= self.class_count_.reshape(-1, 1)

        # Compute means of P(f|c').
        # Flip the one-hot encoding's bits.
        Y_not = np.ones(len(self.classes_), dtype=np.int) - Y
        class_count_not = self.class_count_.sum() - self.class_count_
        self.lambda_not_ = np.dot(Y_not.T, smoothed_f)
        self.lambda_not_ /= class_count_not.reshape(-1, 1)

    def predict(self, X):
        """Predicts the output labels.

        Args:
            X: Input matrix.

        Returns:
            Labeled output vector.
        """
        n_classes = len(self.classes_)
        f = self._compute_smoothed_frequency(X)
        z = np.array([
            self._log_prob(f, self.lambda_[c]).sum(axis=1) -
            self._log_prob(f, self.lambda_not_[c]).sum(axis=1)
            for c in range(n_classes)
        ]).T

        p_c = np.exp(z + self.class_log_prior_)
        p_c_not = 1.0 - np.exp(self.class_log_prior_)
        p = p_c / (p_c + p_c_not)

        return self.classes_[np.argmax(p, axis=1)]

    def _log_prob(self, f, mean):
        """Computes the log probability on poisson distribution.

        Args:
            f: Smoothed frequency.
            mean: Mean.

        Returns:
            Log probability.
        """
        return f * np.log(mean) - mean

    def _compute_smoothed_frequency(self, X, alpha=1.0):
        """Computes the smoothed frequency.

        Args:
            X: Input matrix.
            alpha: Laplace smoothing parameter.

        Returns:
            Smoothed frequency.
        """
        _, n_features = X.shape
        document_lengths = X.sum(axis=1).reshape(-1, 1)
        smoothed_f = (X + alpha) / (document_lengths + alpha * n_features)
        return smoothed_f
