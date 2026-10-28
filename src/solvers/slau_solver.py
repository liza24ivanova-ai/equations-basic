import sys
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decomps.ldlt import ldlt_factor
from decomps.lup import lup_factor
from decomps.qr import qr_factor
from solvers.ldlt_solve import ldlt_solve
from solvers.lup_solve import lup_solve
from solvers.qr_solve import qr_solve


def _read_system(
    stream: TextIO, console: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if console:
        print("Введите матрицу A:", flush=True)

    rows: list[list[str]] = []
    for line in stream:
        if not line.strip():
            break
        rows.append(line.split())

    A: NDArray[np.float64] = np.array(rows, dtype=np.float64)
    if console:
        print("Введите вектор b:", flush=True)
        b: NDArray[np.float64] = np.array(
            [stream.readline() for _ in range(A.shape[0])], dtype=np.float64
        )
    else:
        b: NDArray[np.float64] = np.array(stream.read().split(), dtype=np.float64)

    return A, b


def main(args: list[str]) -> None:
    """
    :param args: list[str], command-line arguments without the program name
    """
    methods: tuple[str, ...] = ("lup", "ldlt", "qr")
    method: str = "lup"
    path: Path | None = None

    if len(args) == 2:
        path = Path(args[0])
        method = args[1]
    elif args:
        if args[0] in methods:
            method = args[0]
        else:
            path = Path(args[0])

    if path is None:
        A, b = _read_system(sys.stdin, console=True)
    else:
        with path.open(encoding="utf-8") as stream:
            A, b = _read_system(stream)

    if method == "lup":
        L, U, p = lup_factor(A)
        x: NDArray = lup_solve(L, U, p, b)
    elif method == "ldlt":
        L, d = ldlt_factor(A)
        x: NDArray = ldlt_solve(L, d, b)
    else:
        F, tau = qr_factor(A)
        x: NDArray = qr_solve(F, tau, b)

    for value in x:
        print(f"{value:.17g}")


if __name__ == "__main__":
    main(sys.argv[1:])
