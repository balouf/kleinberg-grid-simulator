"""Main module."""
import time

from numba import njit  # type: ignore

from kleinberg_grid_simulator.python_implementation.shortcuts import (draw_r_lt_1, draw_r_eq_1, draw_1_lt_r_lt_2,
                                                       draw_r_eq_2, draw_2_lt_r)
from kleinberg_grid_simulator.python_implementation.greedy_walk import edt_gen
from kleinberg_grid_simulator.utils import Result


def python_edt(n=1000, r=2, p=1, q=1, n_runs=10000, numba=True, parallel=False):
    """
    Python-based computation of the expected delivery time (edt).

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
    numba: :class:`bool`, default=True
        Use JiT compilation
    parallel: :class:`bool`, default=False
        Parallelize runs. Use for single, lengthy computation.
        Coarse-grained (high-level) parallelisation is preferred.


    Returns
    -------
    :class:`~kleingrid.kleingrid.EDT`
    """
    start = time.process_time()
    if r < 1:
        gen = draw_r_lt_1(n, r)
    elif r == 1:
        gen = draw_r_eq_1(n)
    elif r < 2:
        gen = draw_1_lt_r_lt_2(n, r)
    elif r == 2:
        gen = draw_r_eq_2(n)
    else:
        gen = draw_2_lt_r(n, r)
    if numba:
        gen = njit(gen)
        main_loop = njit(edt_gen, parallel=parallel)
    else:
        main_loop = edt_gen
    edt = main_loop(gen=gen, n=n, p=p, q=q, n_runs=n_runs)
    return Result(edt=edt, process_time=time.process_time() - start,
                  n=n, r=r, p=p, q=q, n_runs=n_runs, julia=False, numba=numba, parallel=parallel)
