import numpy as np
from sklearn import metrics

from robo.maximizers.base_maximizer import BaseMaximizer
from robo.maximizers.random_sampling import RandomSampling
from robo.initial_design import init_random_uniform


class ApproxSampling(BaseMaximizer):

    def __init__(self, objective_function, lower, upper, pool, distance, replacement, n_samples=500, rng=None):
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
        distance: {"cosine", "euclidean"}
            The distance measurement from
        replacement: bool
            Whether to sample from the pool with replacement

        n_samples: int
            Number of candidates that are samples
        """
        self.pool = pool
        self.distance = distance
        self.random_sampling = RandomSampling(objective_function, lower, upper, n_samples, rng)
        self.replacement = replacement
        super(ApproxSampling, self).__init__(objective_function, lower, upper, rng)

    def _cosine_similarity(self, x, x_):
        """
        Calculates the cosine similarity.
        """
        res = abs(metrics.pairwise.cosine_similarity(np.array(x).reshape(1,-1), np.array(x_).reshape(1,-1)))
        return res

    def _euclidean_distance(self, x, x_):
        """
        Calculates the Euclidean distance.
        """
        res = np.linalg.norm(np.array(x)-np.array(x_))
        return res

    def maximize(self, pool):
        """
        Maximizes the given acquisition function.

        Returns
        -------
        np.ndarray(1,D)
            Point with highest acquisition value.
        """
        self.pool = pool
        x_star_ = self.random_sampling.maximize()

        if self.distance == "cosine":
            calsim = self._cosine_similarity
        elif self.distance == "euclidean":
            calsim = self._euclidean_distance

        id_max = np.argmax(np.array([calsim(x_star_,x_) for x_ in self.pool]))

        x_star = self.pool[id_max]

        if not self.replacement:
            self.pool = np.delete(self.pool, id_max, 0)

        return x_star, pool
