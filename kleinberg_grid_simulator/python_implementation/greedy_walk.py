import numpy as np
from numba import prange  # type: ignore


def edt_gen(gen, n, p, q, n_runs):
    """
    Core computation of Expected Delivery Time (edt).

    Parameters
    ----------
    gen: callable
        Function that draws relative shortcuts
    n: :class:`int`
        Grid siDe
    p: :class:`int`
        Local range
    q: :class:`int`
        Number of shortcuts
    n_runs: :class:`int`
        Number of routes to compute

    Returns
    -------
    :class:`float`
        Expected Delivery Time
    """
    steps = 0
    for _ in prange(n_runs):
        s_x = np.random.randint(n)
        s_y = np.random.randint(n)
        d_x = np.random.randint(n)
        d_y = np.random.randint(n)
        d = np.abs(s_x - d_x) + np.abs(s_y - d_y)
        while d > 0:
            d_s, sh_x, sh_y = 2 * n, -1, -1
            for j in range(q):
                c_s, ch_x, ch_y = 2 * n, -1, -1
                while not ((0 <= ch_x < n - 1) and (0 <= ch_y < n - 1)):
                    r_x, r_y = gen()
                    ch_x = s_x + r_x
                    ch_y = s_y + r_y
                c_s = np.abs(d_x - ch_x) + np.abs(d_y - ch_y)
                if c_s < d_s:
                    d_s, sh_x, sh_y = c_s, ch_x, ch_y
            if d_s < d - p:
                d, s_x, s_y = d_s, sh_x, sh_y
            else:
                d = d - p
                delta_x = min(p, np.abs(d_x - s_x))
                delta_y = p - delta_x
                s_x += delta_x * np.sign(d_x - s_x)
                s_y += delta_y * np.sign(d_y - s_y)
            steps += 1
    return steps / n_runs
