import numpy as np
from numpy.typing import NDArray


def tikhonov(
    A: NDArray, b: NDArray, Gamma: NDArray, x0: NDArray, lam: float
) -> tuple[NDArray, NDArray]:
    """
    :param A: np.array((n, m)), float32 or float64
    :param b: np.array((n,)), same dtype as A
    :param Gamma: np.array((q, m)), same dtype as A
    :param x0: np.array((m,)), same dtype as A
    :param lam: float, lam > 0

    :return: (A_reg, b_reg), same dtype as A
        A_reg - np.array((n + q, m))
        b_reg - np.array((n + q,))
    """
    if not all(np.all(np.isfinite(array)) for array in (A, b, Gamma, x0)):
        raise ArithmeticError(
            "Данные регуляризованной задачи содержат неконечные значения"
        )

    lam_typed: np.floating = A.dtype.type(lam)
    if not np.isfinite(lam_typed):
        raise ArithmeticError(f"Неконечный параметр регуляризации: {lam=}")
    if lam_typed <= 0:
        raise ValueError(f"Параметр регуляризации должен быть положительным: {lam=}")

    A_reg: NDArray = np.vstack((A, lam_typed * Gamma))
    b_reg: NDArray = np.concatenate((b, lam_typed * (Gamma @ x0)))
    if not np.all(np.isfinite(A_reg)) or not np.all(np.isfinite(b_reg)):
        raise ArithmeticError("При регуляризации получены неконечные значения")

    return A_reg, b_reg
