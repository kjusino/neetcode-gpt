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

        # Loss: L = 0.5 * (y_hat - y_true)^2
        loss = 0.5 * ((y_hat - y_true)**2)
        grad_w = np.array(x)
        for i in range(len(grad_w)):
            grad_w[i] = (y_hat - y_true) * y_hat*(1 - y_hat) * x[i]
        grad_b = (y_hat - y_true) * y_hat*(1-y_hat)
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        return (np.round(grad_w, 5), np.round(grad_b, 5)) 
