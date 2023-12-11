import time

from juliacall import Main as jl  # type: ignore
from kleingrid.utils import Result
from kleingrid import __file__ as d
from pathlib import Path

jl.include(str(Path(d).parent / "julia_implementation/kleingrid.jl"))


def big_int_log(n):
    return jl.log2(n)


def julia_edt(n=1000, r=2, p=1, q=1, n_runs=10000):
    """
    Julia-based computation of the expected delivery time (edt).

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

    Returns
    -------
    :class:`~kleingrid.kleingrid.EDT`
    """
    start = time.process_time()
    edt = jl.expected_delivery_time(n, r, p, q, n_runs)
    return Result(edt=edt, process_time=time.process_time() - start,
                  n=n, r=r, p=p, q=q, n_runs=n_runs, julia=True)

