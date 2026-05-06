class Solution:
    """
    A class containing gradient descent implementation for minimizing f(x) = x².
    
    This is an example showing how gradient descent finds the minimum
    of a simple convex quadratic function. The exact minimum is at x = 0.
    """
    
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        """
        Performs gradient descent to minimize the function f(x) = x².
        
        The function finds the value of x that minimizes x² by iteratively
        moving opposite to the gradient (derivative) of the function.
        
        Parameters
        ----------
        iterations : int
            Number of gradient descent steps to perform.
            Must be a non-negative integer.
            - More iterations generally lead to more accurate results
            - Too many iterations may cause floating-point precision issues
            
        learning_rate : float
            Step size for each gradient descent update.
            Controls how much we move opposite to the gradient.
            
            Mathematical constraints for convergence:
            - 0 < learning_rate < 1: Guaranteed convergence to minimum
            - learning_rate = 0.5: Optimal (converges in 1 step for x²)
            - learning_rate = 0: No movement (stuck at initial value)
            - learning_rate > 1: Diverges (moves away from minimum)
            - learning_rate < 0: Moves uphill (finds maximum instead)
            
            Typical values in practice: 0.01 to 0.1
            
        init : int
            Initial guess for the minimizer.
            Can be any integer (positive, negative, or zero).
            
            - If init = 0: Already at minimum (derivative = 0, no updates)
            - Positive values: Will move toward 0 from the right
            - Negative values: Will move toward 0 from the left
            
        Returns
        -------
        float
            The approximate minimizer after the specified number of iterations,
            rounded to 5 decimal places.
            
            For f(x) = x², returns a value close to 0.
            The exact value depends on iterations, learning_rate, and init.
            
        Mathematical Details
        --------------------
        Update rule: x_{k+1} = x_k - η·f'(x_k)
        For f(x) = x²: f'(x) = 2x
        Therefore: x_{k+1} = x_k - η·(2x_k) = x_k(1 - 2η)
        
        Closed-form solution after t iterations:
        x_t = init × (1 - 2η)^t
        
        Examples
        --------
        >>> sol = Solution()
        
        # Basic usage: minimize from init=10 with 3 iterations, η=0.1
        >>> sol.get_minimizer(3, 0.1, 10)
        5.12
        
        # Optimal learning rate achieves minimum in 1 step
        >>> sol.get_minimizer(1, 0.5, 100)
        0.0
        
        # Negative initial values work symmetrically
        >>> sol.get_minimizer(3, 0.1, -10)
        -5.12
        
        # Already at minimum (no movement)
        >>> sol.get_minimizer(100, 0.1, 0)
        0.0
        
        # Slow convergence with small learning rate
        >>> sol.get_minimizer(10, 0.01, 100)
        81.79  # Still far from 0 after 10 iterations
        
        Notes
        -----
        - The function always returns a float rounded to 5 decimal places
        - For non-convergent parameters (η >= 1), results will diverge to ±infinity
        - This is a simplified example; real applications use vectorized operations
        """
        # Initialize minimizer with the starting guess
        minimizer = init
        
        # Perform gradient descent iterations
        for _ in range(iterations):
            # Calculate derivative of f(x) = x² at current point
            derivative = 2 * minimizer
            
            # Update minimizer by moving opposite to gradient
            # The negative sign ensures we move downhill
            minimizer = minimizer - learning_rate * derivative
        
        # Round to 5 decimal places for consistent output
        return round(minimizer, 5)
