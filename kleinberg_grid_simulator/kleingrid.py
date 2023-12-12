from joblib import Parallel, delayed  # type: ignore
from tqdm import tqdm

from kleinberg_grid_simulator.python_implementation.python_edt import python_edt
from kleinberg_grid_simulator.julia_implementation.julia_edt import julia_edt, big_int_log
from kleinberg_grid_simulator.utils import cache_edt_of_r, get_target, gss, get_best_n_values, logger



def compute_edt(n=1000, r=2, p=1, q=1, n_runs=10000, julia=True, numba=True, parallel=False):
    """
    Computation of the expected delivery time (edt).

    Parameters
    ----------
    n: :class:`int`, default=1000
        Grid siDe
    r: :class:`float`, default=2.0
        Shortcut exponent
    p: :class:`int`, default=1
        Local range
    q: :class:`int`, default=1
        Number of shortcuts
    n_runs: :class:`int`, default=10000
        Number of routes to compute
    julia: :class:`bool`, default=True
        Use Julia backend.
    numba: :class:`bool`, default=True
        Use JiT compilation (Python backend)
    parallel: :class:`bool`, default=False
        Parallelize runs (Python backend with Numba). Use for single, lengthy computation.
        Coarse-grained (high-level) parallelisation is preferred.

    Returns
    -------
    :class:`~kleinberg_grid_simulator.utils.Result`
        The expected number of steps to go from one point of the grid to another point of the grid.


    Examples
    --------

    >>> from juliacall import Main as jl
    >>> jl.set_seed(42)
    Julia: TaskLocalRNG()
    >>> from kleinberg_grid_simulator.python_implementation.seed import set_seeds
    >>> set_seeds(42, 51)
    >>> compute_edt(n=1000, r=.5, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=79.93, process_time=..., n=1000, r=0.5, p=1, q=1, n_runs=100, julia=True)
    >>> compute_edt(n=1000, r=.5, n_runs=100, julia=False, numba=False)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=85.93, process_time=..., n=1000, r=0.5, p=1, q=1, n_runs=100, numba=False, parallel=False)
    >>> compute_edt(n=1000, r=1, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=66.39, process_time=..., n=1000, r=1, p=1, q=1, n_runs=100, julia=True)
    >>> compute_edt(n=1000, r=1, n_runs=100, julia=False, numba=False)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=69.44, process_time=..., n=1000, r=1, p=1, q=1, n_runs=100, numba=False, parallel=False)
    >>> compute_edt(n=1000, r=1.5, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=57.91, process_time=..., n=1000, r=1.5, p=1, q=1, n_runs=100, julia=True)
    >>> compute_edt(n=1000, r=1.5, n_runs=100, julia=False, numba=False)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=55.8, process_time=..., n=1000, r=1.5, p=1, q=1, n_runs=100, numba=False, parallel=False)
    >>> compute_edt(n=1000, r=2, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=69.53, process_time=..., n=1000, r=2, p=1, q=1, n_runs=100, julia=True)
    >>> compute_edt(n=1000, r=2, n_runs=100, julia=False, numba=False)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=67.28, process_time=..., n=1000, r=2, p=1, q=1, n_runs=100, numba=False, parallel=False)
    >>> compute_edt(n=1000, r=2.5, n_runs=100)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=187.74, process_time=..., n=1000, r=2.5, p=1, q=1, n_runs=100, julia=True)
    >>> compute_edt(n=1000, r=2.5, n_runs=100, julia=False, numba=False)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Result(edt=167.85, process_time=..., n=1000, r=2.5, p=1, q=1, n_runs=100, numba=False, parallel=False)
    >>> compute_edt(n=10000000000, r=1.5, n_runs=10000, p=2, q=2)  # doctest: +SKIP
    Result(edt=6846.631, process_time=13.0, n=10000000000, r=1.5, p=2, q=2, n_runs=10000, julia=True)
    >>> compute_edt(n=10000000000, r=1.5, n_runs=10000, p=2, q=2, julia=False, parallel=True)  # doctest: +SKIP
    Result(edt=6823.2369, process_time=27.796875, n=10000000000, r=1.5, p=2, q=2, n_runs=10000, numba=True, parallel=True)
    """
    if julia:
        return julia_edt(n=n, r=r, p=p, q=q, n_runs=n_runs)
    else:
        return python_edt(n=n, r=r, p=p, q=q, n_runs=n_runs, numba=numba, parallel=parallel)


def parallelize(values, function=None, n_jobs=-1):
    """
    Straight-forward parallel computing for different values.

    Parameters
    ----------
    values: :class:`list` of :class:`dict`
        Values to test. Each element is a dict of arguments to pass.
    function: callable, default=:meth:`~kleingrid.kleingrid.compute_edt`
        Function to apply.
    n_jobs: :class:`int`, default=-1
        Number of workers to spawn using `joblib convention`_.

    Returns
    -------
    :class:`list`
        The outputs of the function.

    Examples
    --------

    >>> values = [{'n': 2**i, 'n_runs': 100} for i in range(7, 11)]
    >>> res = parallelize(values)
    >>> [r.edt for r in res]  # doctest: +SKIP
    [30.95, 40.82, 54.06, 69.9]

    .. _joblib convention: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    """
    if function is None:
        function = compute_edt

    def with_key(v):
        return function(**v)

    return Parallel(n_jobs=n_jobs)(tqdm((
        delayed(with_key)(v) for v in values), total=len(values)))


def get_bounds(n, offset_start=.1, n_runs=10000, golden_boost=100):
    """
    Parameters
    ----------
    n: :class:`int`, default=1000
        Grid siDe,
    offset_start: :class:`float`, default=.1
        Increments to use to discover upper/lower bounds.
    n_runs: :class:`int`, default=10000
        Number of runs for regular computation.
    golden_boost: class:`int`
        Number of runs multiplier for the Golden-search, which is more noise-sensitive.

    Returns
    -------
    :class:`dict`
        Estimated bounds + relevant input parameters.

    Examples
    --------

    >>> from juliacall import Main as jl
    >>> jl.set_seed(42)
    Julia: TaskLocalRNG()
    >>> get_bounds(2**20, n_runs=100, golden_boost=10)  # doctest: +NORMALIZE_WHITESPACE
    {'n': 1048576, 'n_runs': 100, 'golden_boost': 10, 'ref_edt': 350.54, 'r2+': 2.1828125000000003,
     'r_opt': 1.8667184270002521, 'min_edt': 323.95000000000005, 'r-': 1.825038820250189, 'r2-': 1.4624999999999995}
    """
    f = cache_edt_of_r(compute_edt=compute_edt, n=n, n_runs=n_runs)
    ff = cache_edt_of_r(compute_edt=compute_edt, n=n, n_runs=golden_boost * n_runs)
    r = 2.
    res = {'n': n, 'n_runs': n_runs, 'golden_boost': golden_boost}

    logger.info(f"Computing ref_edt, n={n}")
    ref = f(r)
    res['ref_edt'] = ref

    logger.info(f"Computing upper bound for r2+, n={n}")
    r += offset_start
    while f(r) < 2 * ref:
        r += offset_start
    up2b = r

    logger.info(f"Computing r2+, n={n}")
    up2 = get_target(f, 2, up2b, 2 * ref)
    res['r2+'] = up2

    logger.info(f"Computing lower bound for r-, n={n}")
    r = 2 - offset_start
    while (f(r) < ref) and r >= 0:
        r -= offset_start
    loa = r

    logger.info(f"Computing optimal r in [{loa:.2f}, 2], n={n}")
    m, fm = gss(ff, loa, 2)
    res['r_opt'] = m
    res['min_edt'] = fm

    if r < 0:
        res['r-'] = 0.
        res['r2-'] = 0.
        return res

    logger.info(f"Computing r-, n={n}")
    lo = get_target(f, m, loa, ref)
    res['r-'] = lo

    logger.info(f"Computing lower bound for r2-, n={n}")
    while (f(r) < 2 * ref) and r >= 0:
        r -= offset_start
    lo2a = r

    if lo2a < 0:
        res['r2-'] = 0.
        return res

    logger.info(f"Computing r2-, n={n}")
    b = min(lo2a + offset_start, lo)
    lo2 = get_target(f, b, lo2a, 2 * ref)
    res['r2-'] = lo2

    return res


def estimate_alpha(r, p=10, budget=20):
    """
    Parameters
    ----------
    r: :class:`float`
        Shortcut exponent to investigate.
    p: :class:`int`
        Initial investigation starts at 2**p.
    budget: :class:`int`
        How many seconds we want to spend on the estimation.

    Returns
    -------

    alpha: :class:`float`
        Complexity exponent, e.g. the edt seems to behave in n**alpha.
    n: :class:`int`
        Greatest value of n considered.

    Examples
    --------

    >>> alpha, n = estimate_alpha(1.5, budget=4)
    >>> alpha  # doctest: +SKIP
    0.33442226161905353
    >>> n  # doctest: +SKIP
    10071350
    """
    n1 = 2**p
    while n1 is not None:
        log_of_n = int(100*big_int_log(n1))/100
        logger.info(f"Computing alpha for r={r} between n=2**{log_of_n:.2f} and n=2**{1+log_of_n:.2f}.")
        v1 = compute_edt(n=n1, r=r)
        v2 = compute_edt(n=n1*2, r=r)
        process_time = v1.process_time+v2.process_time
        n1, alpha = get_best_n_values(v1, v2, budget=budget)
        logger.info(f"Estimated alpha={alpha} computed in {process_time:.2f} seconds.")
        if process_time > budget/2/4**alpha:
            break
    return alpha, v2.n
