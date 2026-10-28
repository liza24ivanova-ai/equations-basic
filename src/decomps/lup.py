import numpy as np
from numpy.typing import NDArray


def lup_factor(A: NDArray) -> tuple[NDArray, NDArray, NDArray[np.intp]]:
    """LUP factorization with partial pivoting.

    :param A: np.array((n, n)), float32 or float64, nonsingular

    :raises ZeroDivisionError: if the pivot at some step is exactly zero

    :return: (L, U, p), computed in the dtype of A
        L - np.array((n, n)), unit lower triangular
        U - np.array((n, n)), upper triangular
        p - np.array((n,)), int, row permutation from 0: A[p] == L @ U
    """
    if not np.all(np.isfinite(A)):
        raise ArithmeticError("Матрица A содержит неконечные значения")

    n: int = A.shape[0]
    L: NDArray = np.eye(n, dtype=A.dtype)
    U: NDArray = A.copy()
    p: NDArray[np.intp] = np.arange(n, dtype=np.intp)

    for k in range(n):
        pivot_idx: int = k + int(np.argmax(np.abs(U[k:, k])))
        pivot: np.floating = U[pivot_idx, k]

        if pivot == 0:
            raise ZeroDivisionError(f"Нулевой ведущий элемент на шаге {k=}: {pivot=}")
        if not np.isfinite(pivot):
            raise ArithmeticError(f"Неконечный ведущий элемент на шаге {k=}: {pivot=}")

        if pivot_idx != k:
            U[[k, pivot_idx]] = U[[pivot_idx, k]]
            p[[k, pivot_idx]] = p[[pivot_idx, k]]
            L[[k, pivot_idx], :k] = L[[pivot_idx, k], :k]

        if k < n - 1:
            L[k + 1 :, k] = U[k + 1 :, k] / U[k, k]
            U[k + 1 :, k + 1 :] -= np.outer(L[k + 1 :, k], U[k, k + 1 :])
            U[k + 1 :, k] = 0

    if not np.all(np.isfinite(L)) or not np.all(np.isfinite(U)):
        raise ArithmeticError("При LUP-разложении получены неконечные значения")

    return L, U, p
