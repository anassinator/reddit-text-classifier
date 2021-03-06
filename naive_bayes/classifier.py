# -*- coding: utf-8 -*-


class Classifier(object):

    """Base classifier."""

    def __init__(self):
        """Constructs a Classifier."""
        pass

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Input matrix.
            y: Labeled output vector.
        """
        raise NotImplemented()

    def predict(self, X):
        """Predicts the output labels for the given data.

        Args:
            X: Input matrix.

        Returns:
            Labeled output vector.
        """
        raise NotImplemented()

    def score(self, X, truth):
        """Scores the prediction of the classifier.

        Args:
            X: Input matrix.
            truth: Expected output labels.

        Returns:
            Accuracy.
        """
        y_pred = self.predict(X)
        return sum(y_pred == truth) / len(truth)
