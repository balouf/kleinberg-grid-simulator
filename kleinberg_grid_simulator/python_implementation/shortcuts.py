import numpy as np
from numba import njit  # type: ignore

from kleinberg_grid_simulator.python_implementation.seed import set_seeds


@njit
def radius2shortcut(radius):
    """
    Parameters
    ----------
    radius: :class:`int`
        Radius to draw shortcut from.

    Returns
    -------
    x: :class:`int`
        x relative coordinate
    y: :class:`int`
        y relative coordinate

    Examples
    --------

    >>> set_seeds()
    >>> radius2shortcut(10)
    (-9, 1)
    >>> radius2shortcut(2**40)
    (772882432861, -326629194915)
    """
    angle = np.random.randint(-2 * radius + 1, 2 * radius + 1)
    return (radius - np.abs(angle)), (np.sign(angle) * (radius - np.abs(radius - np.abs(angle))))


def draw_r_lt_1(n, r):
    """
    Parameters
    ----------
    n: :class:`int`
        grid size
    r: :class:`float`
        Shortcut exponent (<1)

    Returns
    -------
    callable
        A shortcut generator

    Examples
    --------

    >>> set_seeds()
    >>> gen = draw_r_lt_1(100, .5)
    >>> gen()
    (0, -103)
    >>> gen = draw_r_lt_1(1000000000, .8)
    >>> gen()
    (1482484865, 59640099)
    """
    expo = 2 - r
    pow_max_radius = (2 * (n - 1) + 1) ** expo - 1

    def generator():
        radius = np.floor((np.random.rand() * pow_max_radius + 1) ** (1 / expo))
        while np.random.rand() * radius * ((1 + 1 / radius) ** expo - 1) > expo:
            radius = np.floor((np.random.rand() * pow_max_radius + 1) ** (1 / expo))
        return radius2shortcut(np.int64(radius))

    return generator


def draw_r_eq_1(n):
    """
    Parameters
    ----------
    n: :class:`int`
        grid size

    Returns
    -------
    callable
        A shortcut generator

    Examples
    --------

    >>> set_seeds()
    >>> gen = draw_r_eq_1(100)
    >>> gen()
    (0, -103)
    >>> gen = draw_r_eq_1(1000000000)
    >>> gen()
    (677037233, 596605187)
    """
    max_radius = 2 * (n - 1) + 1

    def generator(max_radius=max_radius):
        return radius2shortcut(1 + np.random.randint(max_radius))

    return generator


def draw_1_lt_r_lt_2(n, r):
    """
    Parameters
    ----------
    n: :class:`int`
        grid size
    r: :class:`float`
        Shortcut exponent (1<r<2)

    Returns
    -------
    callable
        A shortcut generator

    Examples
    --------

    >>> set_seeds()
    >>> gen = draw_1_lt_r_lt_2(100, 1.5)
    >>> gen()
    (-78, -103)
    >>> gen = draw_1_lt_r_lt_2(1000000000, 1.8)
    >>> gen()
    (123799, 141186)
    """
    expo = 2 - r
    pow_max_radius = (2 * (n - 1)) ** expo - 1
    p1 = 1 / (1 + pow_max_radius / expo)

    def generator(expo=expo, pow_max_radius=pow_max_radius, p1=p1):
        while True:
            if np.random.rand() < p1:
                return radius2shortcut(1)
            else:
                radius = np.ceil((np.random.rand() * pow_max_radius + 1) ** (1 / expo))
                if np.random.rand() * radius * (1 - (1 - 1 / radius) ** expo) < expo:
                    return radius2shortcut(np.int64(radius))

    return generator


def draw_r_eq_2(n):
    """
    Parameters
    ----------
    n: :class:`int`
        grid size

    Returns
    -------
    callable
        A shortcut generator

    Examples
    --------

    >>> set_seeds()
    >>> gen = draw_r_eq_2(100)
    >>> gen()
    (-50, -103)
    >>> gen = draw_r_eq_2(1000000000)
    >>> gen()
    (23, -6)
    """
    max_radius = 2 * (n - 1)
    p1 = 1 / (1 + np.log(max_radius))

    def generator(max_radius=max_radius, p1=p1):
        while True:
            if np.random.rand() < p1:
                return radius2shortcut(1)
            else:
                radius = np.ceil(max_radius ** np.random.rand())
                if np.random.rand() * radius * np.log(1 + 1 / (radius - 1)) < 1:
                    return radius2shortcut(np.int64(radius))

    return generator


def draw_2_lt_r(n, r):
    """
    Parameters
    ----------
    n: :class:`int`
        grid size
    r: :class:`float`
        Shortcut exponent (>2)

    Returns
    -------
    callable
        A shortcut generator

    Examples
    --------

    >>> set_seeds()
    >>> gen = draw_2_lt_r(100, 2.5)
    >>> gen()
    (29, -45)
    >>> gen = draw_2_lt_r(10000000, 2.8)
    >>> gen()
    (2, 0)
    >>> gen = draw_2_lt_r(100000, 2.2)
    >>> gen()
    (0, -1)
    >>> gen()
    (-39, -15)
    """
    expo = r - 2
    pow_max_radius = 1 / (2 * (n - 1)) ** expo - 1
    p1 = 1 / (1 - pow_max_radius / expo)

    def generator(expo=expo, pow_max_radius=pow_max_radius, p1=p1):
        while True:
            if np.random.rand() < p1:
                return radius2shortcut(1)
            else:
                radius = np.ceil(1 / (np.random.rand() * pow_max_radius + 1) ** (1 / expo))
                if np.random.rand() * radius * ((1 + 1 / (radius - 1)) ** expo - 1) < expo:
                    return radius2shortcut(np.int64(radius))

    return generator
