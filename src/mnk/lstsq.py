import numpy as np
from numpy.typing import NDArray

from decomps.ldlt import ldlt_factor
from decomps.lup import lup_factor
from decomps.qr import apply_q, qr_factor
from solvers.ldlt_solve import ldlt_solve
from solvers.lup_solve import lup_solve
from solvers.upper_solve import upper_solve


def lstsq(A: NDArray, b: NDArray, method: str = "qr") -> NDArray:
    """
    :param A: np.array((n, m)), float32 or float64, n >= m >= 1, rank m
    :param b: np.array((n,)), same dtype as A
    :param method: str -- "qr", "normal_ldlt" or "normal_lup"

    :return: np.array((m,)), computed in the dtype of A -- x
    """
    if method not in ("qr", "normal_ldlt", "normal_lup"):
        raise ValueError(f"Неизвестный метод МНК: {method=}")
    if not all(np.all(np.isfinite(array)) for array in (A, b)):
        raise ArithmeticError("Матрица A или вектор b содержат неконечные значения")

    if method == "qr":
        m: int = A.shape[1]
        F, tau = qr_factor(A)
        c: NDArray = apply_q(F, tau, b, transpose=True)
        return upper_solve(F[:m], c[:m])

    G: NDArray = A.T @ A
    c: NDArray = A.T @ b
    if method == "normal_ldlt":
        L, d = ldlt_factor(G)
        return ldlt_solve(L, d, c)

    L, U, p = lup_factor(G)
    return lup_solve(L, U, p, c)
