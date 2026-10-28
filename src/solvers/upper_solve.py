import numpy as np
from numpy.typing import NDArray


def upper_solve(U: NDArray, b: NDArray) -> NDArray:
    """
    :param U: np.array((n, n)), float32 or float64, nonsingular upper triangular
    :param b: np.array((n,)), same dtype as U

    :return: np.array((n,)), same dtype as U -- x
    """
    if not all(np.all(np.isfinite(array)) for array in (U, b)):
        raise ArithmeticError("Матрица U или вектор b содержат неконечные значения")

    n: int = U.shape[0]
    x: NDArray = np.zeros(n, dtype=U.dtype)

    for i in range(n - 1, -1, -1):
        row_sum: np.floating = U.dtype.type(U[i, i + 1 :] @ x[i + 1 :])
        if U[i, i] == 0:
            raise ZeroDivisionError(
                f"Нулевой диагональный элемент на шаге {i=}: {U[i, i]=}"
            )
        x[i] = (b[i] - row_sum) / U[i, i]

    if not np.all(np.isfinite(x)):
        raise ArithmeticError("При обратной подстановке получены неконечные значения")

    return x
