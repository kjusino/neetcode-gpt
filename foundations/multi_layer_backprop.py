import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        # L(x) = f(g(h(i(x)))) 
        # L(x) = mse(W2,b2(ReLU(W1,b1(x))))

        #types
        x     = np.array(x,      dtype=float)
        W1    = np.array(W1,     dtype=float)
        b1    = np.array(b1,     dtype=float)
        W2    = np.array(W2,     dtype=float)
        b2    = np.array(b2,     dtype=float)
        y_true = np.array(y_true, dtype=float)

        #forwards
        # L(x) = mse(W2(ReLU(W1(x))))
        z1 = np.dot(x, W1.T) + b1 #W1(x)
        a1 = np.maximum(0.0, z1) #ReLU(W1(x))
        z2 = np.dot(a1, W2.T) + b2 #W2(ReLU(W1(x)))
        L = np.mean((z2 - y_true)**2) #mse(W2(ReLU(W1(x))))

        #backwards
        # ∇L
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = 2 * (z2 - y_true) / n
        dW2 = np.outer(dz2.T, a1)
        db2 = dz2
        da1 = np.dot(W2.T, db2)
        dz1 = da1 * np.where(z1 > 0, 1.0, 0.0)
        dW1 = np.outer(dz1.T, x)
        db1 = dz1

        return {
            'loss': np.round(L, 4),
            'dW1': np.round(dW1+0.0, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
