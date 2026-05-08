import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        y_pred = y_pred + 0.0000001
        ans = (-1/len(y_true)) * np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
        return np.round(ans, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        y_pred = y_pred + 0.0000001
        ans = (-1/len(y_true)) * np.sum(y_true * np.log(y_pred))
        return np.round(ans, 4)
