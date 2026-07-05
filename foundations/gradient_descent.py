class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        if iterations < 1:
            return init
        x_cur = (iterations+1)*[0]
        for i in range(iterations+1):
            if i==0:
                x_cur[i] = init
            else:
                x_cur[i] = x_cur[i-1] - learning_rate * 2 * x_cur[i-1]

        return round(x_cur[iterations], 5)