from dataclasses import dataclass
from functools import cache
from typing import Optional
import logging

import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


@dataclass
class Result:
    """
    Dataclass to represent the results.
    """
    edt: float
    process_time: float
    n: int
    r: float
    p: int
    q: int
    n_runs: int
    julia: bool
    numba: bool = True
    parallel: bool = False

    def __repr__(self):
        exclude = {'numba', 'parallel'} if self.julia else {'julia'}
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items() if key not in exclude]
        return "{}({})".format(type(self).__name__, ", ".join(kws))


def cache_edt_of_r(compute_edt, n=10000, n_runs=10000, **kwargs):
    """
    Parameters
    ----------
    compute_edt: callable
        Function that computes... EDT!
    n: :class:`int`, default=10000
        Grid siDe
    n_runs: :class:`int`, default=10000
        Number of routes to compute
    kwargs: :class:`dict`
        Other parameters

    Returns
    -------
    callable
        A cached function that computes the edt as a function of r.
    """
    def f(r):
        return compute_edt(r=r, n=n, n_runs=n_runs, **kwargs).edt
    return cache(f)


def get_target(f, a, b, t):
    """
    Solve by dichotomy f(x)=t

    Parameters
    ----------
    f: callable
        f is monotonic between a and b, possibly noisy.
    a: :class:`float`
        f(a) < t
    b: :class:`float`
        f(b) > t
    t: :class:`float`
        Target

    Returns
    -------
    :class:`float`
        The (possibly approximated) solution of f(x)=t

    Examples
    --------

    >>> f = cache(lambda x: (x-2)**2)
    >>> x = get_target(f, 2., 10., 2.)
    >>> f"{x:.4f}"
    '3.4142'
    """
    fa = f(a)
    fb = f(b)
    c = (a+b)/2
    fc = f(c)
    while fa < fc < fb:
        if fc < t:
            a, fa = c, fc
        else:
            b, fv = c, fc
        c = (a+b)/2
        fc = f(c)
    logger.info("Noise limit reached.")
    return c


def gss(f, a, b, tol=1e-5):
    """
    Find by Golden-section search the minimum of a function f.

    Parameters
    ----------
    f: callable
        f, possibly noisy, is convex on [a, b].
    a: :class:`float`
        Left guess.
    b: :class:`float`
        Right guess.
    tol: :class:`float`
        Exit thresold on x.

    Returns
    -------
    :class:`float`
        The (possibly approximated) value that minimizes f over [a, b].
    :class:`float`
        The (possibly approximated) minimum of f over [a, b].

    Examples
    --------
    >>> f = cache(lambda x: (x-2)**2)
    >>> x = gss(f, 1, 5)
    >>> f"f({x[0]:.4f}) = {x[1]:.4f}"
    'f(2.0000) = 0.0000'

    """
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    while abs(b - a) > tol:
        logger.info(f"Optimal between {a:.2f} and {b:.2f}")
        if f(c) < f(d):
            b = d
            d = c
            c = b - (b - a) / gr
            if f(c) > f(a):
                logger.info("Noise limit reached.")
                break
        else:
            a = c
            c = d
            d = a + (b - a) / gr
            if f(d) > f(b):
                logger.info("Noise limit reached.")
                break
    return (c + d) / 2, (f(c)+f(d))/2


def get_alpha(v1, v2):
    gap = 1 # int(big_int_log(v2.n)-big_int_log(v1.n))
    return (np.log2(v2.edt)-np.log2(v1.edt))/gap


def get_best_n_values(v1, v2, budget=20):
    n1: Optional[int] = None
    alpha = get_alpha(v1, v2)
    c = v2.process_time / (v2.n)**alpha
    n1 = int((budget/(1+2**alpha)/c)**(1/alpha))
    if n1 <= 2*v1.n:
        n1 = None
    return n1, alpha
