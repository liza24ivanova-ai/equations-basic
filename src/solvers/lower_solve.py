import numpy as np
from numpy.typing import NDArray


def lower_solve(L: NDArray, b: NDArray) -> NDArray:
    """
    :param L: np.array((n, n)), float32 or float64, nonsingular lower
              triangular, diagonal not necessarily unit
    :param b: np.array((n,)), same dtype as L

    :return: np.array((n,)), same dtype as L -- x
    """
    if not all(np.all(np.isfinite(array)) for array in (L, b)):
        raise ArithmeticError("Матрица L или вектор b содержат неконечные значения")

    n: int = L.shape[0]
    x: NDArray = np.zeros(n, dtype=L.dtype)

    for i in range(n):
        row_sum: np.floating = L.dtype.type(L[i, :i] @ x[:i])
        if L[i, i] == 0:
            raise ZeroDivisionError(
                f"Нулевой диагональный элемент на шаге {i=}: {L[i, i]=}"
            )
        x[i] = (b[i] - row_sum) / L[i, i]

    if not np.all(np.isfinite(x)):
        raise ArithmeticError("При прямой подстановке получены неконечные значения")

    return x
