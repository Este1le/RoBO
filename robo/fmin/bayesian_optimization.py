import logging
import george
import numpy as np

from pybnn.dngo import DNGO

from robo.priors.default_priors import DefaultPrior
from robo.models.wrapper_bohamiann import WrapperBohamiann
from robo.models.gaussian_process import GaussianProcess
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.models.random_forest import RandomForest
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.maximizers.random_sampling import RandomSampling
from robo.maximizers.differential_evolution import DifferentialEvolution
from robo.maximizers.exact_sampling import ExactSampling
from robo.maximizers.approx_sampling import ApproxSampling
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.pi import PI
from robo.acquisition_functions.log_ei import LogEI
from robo.acquisition_functions.lcb import LCB
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.initial_design.init_latin_hypercube_sampling import init_latin_hypercube_sampling
from robo.initial_design.init_exact_random import init_exact_random


logger = logging.getLogger(__name__)


def bayesian_optimization(objective_function, lower, upper, num_iterations=30, X_init=None, Y_init=None,
                          maximizer="random", acquisition_func="log_ei", model_type="gp_mcmc",
                          n_init=3, rng=None, output_path=None,
                          sampling_method="origin", distance="cosine", replacement=True, pool=None, best=19.490554):
    """
    General interface for Bayesian optimization for global black box
    optimization problems.

    Parameters
    ----------
    objective_function: function
        The objective function that is minimized. This function gets a numpy
        array (D,) as input and returns the function value (scalar)
    lower: np.ndarray (D,)
        The lower bound of the search space
    upper: np.ndarray (D,)
        The upper bound of the search space
    num_iterations: int
        The number of iterations (initial design + BO)
    X_init: np.ndarray(N,D)
            Initial points to warmstart BO
    Y_init: np.ndarray(N,1)
            Function values of the already initial points
    maximizer: {"random", "scipy", "differential_evolution"}
        The optimizer for the acquisition function.
    acquisition_func: {"ei", "log_ei", "lcb", "pi"}
        The acquisition function
    model_type: {"gp", "gp_mcmc", "rf", "bohamiann", "dngo"}
        The model for the objective function.
    n_init: int
        Number of points for the initial design. Make sure that it
        is <= num_iterations.
    output_path: string
        Specifies the path where the intermediate output after each iteration will be saved.
        If None no output will be saved to disk.
    rng: numpy.random.RandomState
        Random number generator
    sampling_method: {"origin", "approx", "exact"}
        Specify the method to choose next sample to update model.
        approx: choose the sample in the candidate pool that is closest (measured by distance
        arg) to the one returned from maximizing acquisition function.
        exact: evaluate all samples in the candidate pool on acquisition function
        and choose the one with maximum output.
    distance: {"cosine", "euclidean"}
        The distance measurement for approximation sampling.
    replacement: boolean
        Whether to sample from pool with replacement. (approximation sampling)
    pool: np.ndarray(N,D)
        Candidate pool containing possible x
    best: float
        Stop training when the best point is sampled.
    Returns
    -------
        dict with all results
    """
    assert upper.shape[0] == lower.shape[0], "Dimension miss match"
    assert np.all(lower < upper), "Lower bound >= upper bound"
    assert n_init <= num_iterations, "Number of initial design point has to be <= than the number of iterations"

    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    cov_amp = 2
    n_dims = lower.shape[0]

    initial_ls = np.ones([n_dims])
    exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                               ndim=n_dims)
    kernel = cov_amp * exp_kernel

    prior = DefaultPrior(len(kernel) + 1)

    n_hypers = 3 * len(kernel)
    if n_hypers % 2 == 1:
        n_hypers += 1

    if model_type == "gp":
        model = GaussianProcess(kernel, prior=prior, rng=rng,
                                normalize_output=False, normalize_input=True,
                                lower=lower, upper=upper)
    elif model_type == "gp_mcmc":
        model = GaussianProcessMCMC(kernel, prior=prior,
                                    n_hypers=n_hypers,
                                    chain_length=200,
                                    burnin_steps=100,
                                    normalize_input=True,
                                    normalize_output=False,
                                    rng=rng, lower=lower, upper=upper)

    elif model_type == "rf":
        model = RandomForest(rng=rng)

    elif model_type == "bohamiann":
        model = WrapperBohamiann()

    elif model_type == "dngo":
        model = DNGO()

    else:
        raise ValueError("'{}' is not a valid model".format(model_type))

    if acquisition_func == "ei":
        a = EI(model)
    elif acquisition_func == "log_ei":
        a = LogEI(model)
    elif acquisition_func == "pi":
        a = PI(model)
    elif acquisition_func == "lcb":
        a = LCB(model)
    else:
        raise ValueError("'{}' is not a valid acquisition function"
                         .format(acquisition_func))

    if model_type == "gp_mcmc":
        acquisition_func = MarginalizationGPMCMC(a)
    else:
        acquisition_func = a

    if maximizer == "random":
        max_func = RandomSampling(acquisition_func, lower, upper, rng=rng)
    elif maximizer == "scipy":
        max_func = SciPyOptimizer(acquisition_func, lower, upper, rng=rng)
    elif maximizer == "differential_evolution":
        max_func = DifferentialEvolution(acquisition_func, lower, upper, rng=rng)
    else:
        raise ValueError("'{}' is not a valid function to maximize the "
                         "acquisition function".format(maximizer))

    if sampling_method == "exact":
        max_func = ExactSampling(acquisition_func, lower, upper, pool, rng=rng)
        init_design, pool = init_exact_random
    elif sampling_method == "approx":
        max_func = ApproxSampling(acquisition_func, lower, upper, pool, distance, replacement, rng=rng)
        init_design, pool = init_exact_random
    else:
        init_design, pool = init_latin_hypercube_sampling

    bo = BayesianOptimization(objective_function, lower, upper,
                              acquisition_func, model, max_func, pool, best,
                              sampling_method, distance, replacement,
                              initial_points=n_init, rng=rng,
                              initial_design=init_design,
                              output_path=output_path)

    x_best, f_min = bo.run(num_iterations, X=X_init, y=Y_init)

    results = dict()
    results["x_opt"] = x_best
    results["f_opt"] = f_min
    results["incumbents"] = [inc for inc in bo.incumbents]
    results["incumbent_values"] = [val for val in bo.incumbents_values]
    results["runtime"] = bo.runtime
    results["overhead"] = bo.time_overhead
    results["X"] = [x.tolist() for x in bo.X]
    results["y"] = [y for y in bo.y]
    return results
