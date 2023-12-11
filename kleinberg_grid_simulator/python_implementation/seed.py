import numpy as np
from numba import njit # type: ignore


@njit
def set_numba_seed(seed):
    np.random.seed(seed)


def set_seeds(seed=42, numba_seed=None):
    """
    Parameters
    ----------
    seed: :class:`int`, optional
        Numpy seed
    numba_seed: :class:`int`, optional
        Numba seed

    Returns
    -------
    None
    """
    if numba_seed is None:
        numba_seed = seed
    np.random.seed(seed)
    set_numba_seed(numba_seed)
