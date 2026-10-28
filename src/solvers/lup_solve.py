import numpy as np
from numpy.typing import NDArray

from solvers.lower_solve import lower_solve
from solvers.upper_solve import upper_solve


def lup_solve(L: NDArray, U: NDArray, p: NDArray[np.intp], b: NDArray) -> NDArray:
    """
    :param L, U, p: returned by lup_factor, must not be modified
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """
    y: NDArray = lower_solve(L, b[p])
    return upper_solve(U, y)
