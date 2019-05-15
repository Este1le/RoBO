import numpy as np


def init_exact_random(lower, upper, n_points, pool, replacement, rng=None):
    """
    Samples N data points uniformly.

    Parameters
    ----------
    lower: np.ndarray (D)
        Lower bounds of the input space
    upper: np.ndarray (D)
        Upper bounds of the input space
    n_points: int
        The number of initial data points
    pool: np.ndarray (N,D)
        The candidate pool
    replacement: Boolean
        Whether to sample from the pool with replacement
    rng: np.random.RandomState
            Random number generator
    Returns
    -------
    init: np.ndarray(N,D)
        The initial design data points
    index: np.ndarray(N,1)
        The index of initial design data points
    """
    index = np.random.choice(np.arange(pool.shape[0]), n_points, replace=False)
    init = pool[index]
    if not replacement:
        pool = np.delete(pool, index, 0)

    return init, pool
