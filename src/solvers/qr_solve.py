from numpy.typing import NDArray

from decomps.qr import apply_q
from solvers.upper_solve import upper_solve


def qr_solve(F: NDArray, tau: NDArray, b: NDArray) -> NDArray:
    """
    :param F, tau: returned by qr_factor (square A), must not be modified
    :param b: np.array((n,)), same dtype as F

    :return: np.array((n,)), same dtype as F -- x
    """
    y: NDArray = apply_q(F, tau, b, transpose=True)
    return upper_solve(F, y)
