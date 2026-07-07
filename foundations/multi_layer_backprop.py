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
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        z1 = np.dot(W1,x)+b1
        a1 =[x  if x>0 else 0 for x in z1]
        z2 = np.dot(W2, a1)+b2
        MSE = np.round(np.mean((z2-y_true)**2),4)
        relu_mask = np.array([1 if x > 0 else 0 for x in z1])
            
        db2 = np.round(2 * (z2 - y_true) / len(y_true), 4)
        dW2 = np.round(np.outer(db2, a1), 4)
        
        da1 = np.dot(db2, W2)
        dz1 = da1 * relu_mask
        db1 = np.round(dz1, 4)
        dW1 = np.round(np.outer(db1, x), 4)
        
        return {'loss': float(MSE), 'dW1': dW1.tolist(), 'db1': db1.tolist(), 'dW2': dW2.tolist(), 'db2': db2.tolist()}