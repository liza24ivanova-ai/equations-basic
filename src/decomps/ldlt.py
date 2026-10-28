import numpy as np
from numpy.typing import NDArray


def ldlt_factor(A: NDArray) -> tuple[NDArray, NDArray]:
    """LDL^T factorization without pivoting.

    :param A: np.array((n, n)), float32 or float64, symmetric

    :raises ZeroDivisionError: if some d_k is exactly zero

    :return: (L, d), computed in the dtype of A
        L - np.array((n, n)), unit lower triangular
        d - np.array((n,)), A == L @ np.diag(d) @ L.T
    """
    if not np.all(np.isfinite(A)):
        raise ArithmeticError("Матрица A содержит неконечные значения")

    n: int = A.shape[0]
    L: NDArray = np.eye(n, dtype=A.dtype)
    d: NDArray = np.zeros(n, dtype=A.dtype)

    for k in range(n):
        L_d: NDArray = L[k, :k] * d[:k]
        d[k] = A[k, k] - L[k, :k] @ L_d

        if d[k] == 0:
            raise ZeroDivisionError(
                f"Нулевой диагональный элемент на шаге {k=}: {d[k]=}"
            )
        if not np.isfinite(d[k]):
            raise ArithmeticError(
                f"Неконечный диагональный элемент на шаге {k=}: {d[k]=}"
            )

        if k < n - 1:
            L[k + 1 :, k] = (A[k + 1 :, k] - L[k + 1 :, :k] @ L_d) / d[k]

    if not np.all(np.isfinite(L)):
        raise ArithmeticError("При LDLT-разложении получены неконечные значения")

    return L, d
