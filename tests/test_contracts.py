import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

from decomps.ldlt import ldlt_factor
from decomps.lup import lup_factor
from decomps.qr import apply_q, form_q, qr_factor
from mnk.lstsq import lstsq
from mnk.tikhonov import tikhonov
from solvers.ldlt_solve import ldlt_solve
from solvers.lower_solve import lower_solve
from solvers.lup_solve import lup_solve
from solvers.qr_solve import qr_solve
from solvers.upper_solve import upper_solve

FACTORIZATIONS = {"lup": lup_factor, "ldlt": ldlt_factor, "qr": qr_factor}


def _solve(method: str, factors: tuple[NDArray, ...], b: NDArray) -> NDArray:
    if method == "lup":
        return lup_solve(factors[0], factors[1], factors[2], b)
    if method == "ldlt":
        return ldlt_solve(factors[0], factors[1], b)
    return qr_solve(factors[0], factors[1], b)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", FACTORIZATIONS)
def test_factor_solve_preserves_dtype_and_readonly_inputs(
    dtype: type[np.floating], method: str
) -> None:
    rng = np.random.default_rng(42)
    M: NDArray = rng.normal(size=(6, 6)).astype(dtype)
    A: NDArray = M.T @ M + dtype(6) * np.eye(6, dtype=dtype)
    x_true: NDArray = rng.normal(size=6).astype(dtype)
    b: NDArray = A @ x_true
    A.setflags(write=False)
    b.setflags(write=False)

    factors = FACTORIZATIONS[method](A)
    for i, factor in enumerate(factors):
        expected_dtype = np.intp if method == "lup" and i == 2 else dtype
        assert factor.dtype == expected_dtype
        factor.setflags(write=False)

    tol: float = float(100 * np.finfo(dtype).eps)
    for rhs, expected in ((b, x_true), (dtype(2) * b, dtype(2) * x_true)):
        x: NDArray = _solve(method, factors, rhs)
        assert x.dtype == dtype
        assert_allclose(x, expected, rtol=tol, atol=tol)

    if method == "qr":
        F, tau = factors[0], factors[1]
        Q: NDArray = form_q(F, tau)
        y: NDArray = apply_q(F, tau, b)
        assert Q.dtype == y.dtype == dtype
        assert_allclose(Q.T @ Q, np.eye(6), rtol=tol, atol=tol)
        assert_allclose(y, Q @ b, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["lower", "upper"])
def test_triangular_solve_preserves_dtype_and_readonly_inputs(
    dtype: type[np.floating], kind: str
) -> None:
    A: NDArray = np.array([[2, 1, -1], [1, 3, 2], [-2, 1, 4]], dtype=dtype)
    T: NDArray = np.tril(A) if kind == "lower" else np.triu(A)
    x_true: NDArray = np.array([1, -2, 3], dtype=dtype)
    b: NDArray = T @ x_true
    T.setflags(write=False)
    b.setflags(write=False)

    x: NDArray = lower_solve(T, b) if kind == "lower" else upper_solve(T, b)
    assert x.dtype == dtype
    assert_array_equal(x, x_true)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["qr", "normal_ldlt", "normal_lup"])
def test_lstsq_preserves_dtype_and_readonly_inputs(
    dtype: type[np.floating], method: str
) -> None:
    rng = np.random.default_rng(43)
    A: NDArray = rng.normal(size=(20, 5)).astype(dtype)
    b: NDArray = rng.normal(size=20).astype(dtype)
    A.setflags(write=False)
    b.setflags(write=False)

    x: NDArray = lstsq(A, b, method=method)
    x_ref: NDArray = np.linalg.lstsq(
        A.astype(np.float64), b.astype(np.float64), rcond=None
    )[0]
    tol: float = float(100 * np.finfo(dtype).eps)
    assert x.dtype == dtype
    assert_allclose(x, x_ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tikhonov_preserves_dtype_inputs_and_objective(
    dtype: type[np.floating],
) -> None:
    rng = np.random.default_rng(44)
    A: NDArray = rng.normal(size=(4, 3)).astype(dtype)
    b: NDArray = rng.normal(size=4).astype(dtype)
    Gamma: NDArray = rng.normal(size=(2, 3)).astype(dtype)
    x0: NDArray = rng.normal(size=3).astype(dtype)
    x: NDArray = rng.normal(size=3).astype(dtype)
    lam: np.float64 = np.float64(0.3)
    for array in (A, b, Gamma, x0):
        array.setflags(write=False)

    A_reg, b_reg = tikhonov(A, b, Gamma, x0, lam)
    assert A_reg.dtype == b_reg.dtype == dtype
    lhs: np.floating = np.linalg.norm(A_reg @ x - b_reg) ** 2
    rhs: np.floating = (
        np.linalg.norm(A @ x - b) ** 2 + lam**2 * np.linalg.norm(Gamma @ (x - x0)) ** 2
    )
    assert lhs == pytest.approx(rhs, rel=100 * np.finfo(dtype).eps)


@pytest.mark.parametrize(
    "dtype,exponent",
    [(np.float32, -80), (np.float32, 80), (np.float64, -600), (np.float64, 600)],
)
def test_qr_handles_extreme_scales(dtype: type[np.floating], exponent: int) -> None:
    A: NDArray = np.array([[1, 2], [-3, 1], [2, -2]], dtype=dtype)
    scale: np.floating = dtype(np.ldexp(1.0, exponent))
    F, tau = qr_factor(A * scale)
    Q: NDArray = form_q(F, tau)
    R: NDArray = np.triu(F[:2]) / scale
    tol: float = float(100 * np.finfo(dtype).eps)
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(tau))
    assert_allclose(Q @ R, A, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", FACTORIZATIONS)
def test_small_nonzero_pivots_are_not_discarded(
    dtype: type[np.floating], method: str
) -> None:
    A: NDArray = np.array([[np.finfo(dtype).tiny]], dtype=dtype)
    b: NDArray = A[:, 0].copy()
    x: NDArray = _solve(method, FACTORIZATIONS[method](A), b)
    assert_array_equal(x, np.ones(1, dtype=dtype))


@pytest.mark.parametrize("method", FACTORIZATIONS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_factorizations_report_nonfinite_input(method: str, value: float) -> None:
    with pytest.raises(ArithmeticError):
        FACTORIZATIONS[method](np.array([[value]]))


@pytest.mark.parametrize("method", FACTORIZATIONS)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_solvers_report_nonfinite_rhs(method: str, value: float) -> None:
    factors = FACTORIZATIONS[method](np.eye(2))
    with pytest.raises(ArithmeticError):
        _solve(method, factors, np.array([value, 1.0]))


@pytest.mark.parametrize("method", FACTORIZATIONS)
def test_solvers_report_singular_systems(method: str) -> None:
    A: NDArray = np.diag([1.0, 0.0])
    with pytest.raises(ArithmeticError):
        _solve(method, FACTORIZATIONS[method](A), np.ones(2))


@pytest.mark.parametrize("value", [0.0, np.nan, np.inf, -np.inf])
def test_ldlt_solve_reports_invalid_diagonal(value: float) -> None:
    with pytest.raises(ArithmeticError):
        ldlt_solve(np.eye(2), np.array([1.0, value]), np.ones(2))


@pytest.mark.parametrize("kind", ["lower", "upper"])
def test_triangular_solve_reports_overflow(kind: str) -> None:
    solver = lower_solve if kind == "lower" else upper_solve
    with pytest.raises(ArithmeticError):
        solver(np.array([[1e-300]]), np.array([1e300]))


def test_lup_reports_elimination_overflow() -> None:
    with pytest.raises(ArithmeticError):
        lup_factor(np.array([[1.0, 1e308], [-1.0, 1e308]]))


def test_ldlt_handles_negative_diagonal_without_pivoting() -> None:
    A: NDArray = np.array([[1.0, 2.0], [2.0, 1.0]])
    L, d = ldlt_factor(A)
    assert np.any(d < 0)
    assert_allclose((L * d) @ L.T, A, rtol=0, atol=0)
