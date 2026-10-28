import numpy as np
from numpy.typing import NDArray


def qr_factor(A: NDArray) -> tuple[NDArray, NDArray]:
    """Householder QR factorization in compact form.

    :param A: np.array((n, m)), float32 or float64, n >= m >= 1

    :return: (F, tau), computed in the dtype of A
        F   - np.array((n, m)), R on and above the diagonal, reflectors below
        tau - np.array((min(m, n - 1),))
    """
    if not np.all(np.isfinite(A)):
        raise ArithmeticError("Матрица A содержит неконечные значения")

    n, m = A.shape
    F: NDArray = A.copy()
    tau: NDArray = np.zeros(min(m, n - 1), dtype=A.dtype)

    for k in range(tau.size):
        v: NDArray = F[k:, k].copy()
        scale: np.floating = np.max(np.abs(v))
        if scale == 0:
            continue

        v /= scale
        v_norm: np.floating = np.linalg.norm(v)
        beta: np.floating = -v_norm if v[0] >= 0 else v_norm
        v[0] -= beta
        v /= v[0]
        tau[k] = A.dtype.type(2) / (v @ v)

        if k < m - 1:
            w: NDArray = tau[k] * (v @ F[k:, k + 1 :])
            F[k:, k + 1 :] -= np.outer(v, w)

        F[k, k] = beta * scale
        F[k + 1 :, k] = v[1:]

    if not np.all(np.isfinite(F)) or not np.all(np.isfinite(tau)):
        raise ArithmeticError("При QR-разложении получены неконечные значения")

    return F, tau


def apply_q(F: NDArray, tau: NDArray, b: NDArray, transpose: bool = False) -> NDArray:
    """
    :param F, tau: returned by qr_factor
    :param b: np.array((n,)), same dtype as F
    :param transpose: bool -- False: Q b, True: Q^T b

    :return: np.array((n,)), same dtype as F
    """
    if not all(np.all(np.isfinite(array)) for array in (F, tau, b)):
        raise ArithmeticError("Факторы QR или вектор b содержат неконечные значения")

    y: NDArray = b.copy()
    indices: range = range(tau.size) if transpose else range(tau.size - 1, -1, -1)

    for k in indices:
        if tau[k] == 0:
            continue

        v: NDArray = np.ones(F.shape[0] - k, dtype=F.dtype)
        v[1:] = F[k + 1 :, k]
        y[k:] -= v * (tau[k] * (v @ y[k:]))

    if not np.all(np.isfinite(y)):
        raise ArithmeticError("При применении Q получены неконечные значения")

    return y


def form_q(F: NDArray, tau: NDArray, thin: bool = True) -> NDArray:
    """
    :param F, tau: returned by qr_factor
    :param thin: bool -- True: Q_1 of shape (n, m), False: Q of shape (n, n)

    :return: np.array, same dtype as F
    """
    if not all(np.all(np.isfinite(array)) for array in (F, tau)):
        raise ArithmeticError("Факторы QR содержат неконечные значения")

    n, m = F.shape
    Q: NDArray = np.eye(n, m if thin else n, dtype=F.dtype)

    for k in range(tau.size - 1, -1, -1):
        if tau[k] == 0:
            continue

        v: NDArray = np.ones(n - k, dtype=F.dtype)
        v[1:] = F[k + 1 :, k]
        Q[k:] -= np.outer(v, tau[k] * (v @ Q[k:]))

    if not np.all(np.isfinite(Q)):
        raise ArithmeticError("При построении Q получены неконечные значения")

    return Q
