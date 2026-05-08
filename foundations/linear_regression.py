import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        pred = np.zeros(len(X))
        for i in range(len(X)):
            pred[i] = np.round(np.dot(X[i], weights), 5)
        return pred

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        ans = 1/len(model_prediction) * np.sum((model_prediction - ground_truth)**2)
        return np.round(ans, 5)
