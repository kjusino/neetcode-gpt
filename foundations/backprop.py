import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        z = np.dot(x, w) + b
        y_hat = 1/(1 + np.exp(-z))

        error = y_hat - y_true
        sigmoid_derivative = y_hat*(1 - y_hat)
        delta = error * sigmoid_derivative

        grad_w = delta * x
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        return (np.round(grad_w, 5), np.round(delta, 5)) 
