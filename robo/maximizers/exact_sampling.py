import numpy as np

from robo.maximizers.base_maximizer import BaseMaximizer
from robo.initial_design import init_random_uniform


class ExactSampling(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, n_samples=500, rng=None):
        """
        Samples rest candidates in the candidate pool and returns the point with the highest objective value.

        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        pool: np.ndarray(N,D)
            Candidate pool containing possible x
        n_samples: int
            Number of candidates that are samples
        """
        self.n_samples = n_samples
        super(ExactSampling, self).__init__(objective_function, lower, upper, rng)

    def maximize(self, pool=None):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """
        X = pool
        y = self.objective_func(X)
        x_star = X[y.argmax()]

        pool = np.delete(pool, y.argmax(), 0)

        return x_star, pool
