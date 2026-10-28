# Repository Guidelines

## Project Structure & Module Organization

This repository contains a numerical linear algebra lab. Put implementations in `src/`: matrix factorizations in `decomps/`, linear-system solvers and the command-line solver in `solvers/`, and least-squares and Tikhonov routines in `mnk/`. The top-level modules cover conditioning, accuracy, regularization, and the reservoir problem. Public tests live in `tests/test_public.py`; `pytest.ini` adds `src` to the import path, so imports use, for example, `from decomps.lup import lup_factor`.

## Build, Test, and Development Commands

Run these commands from the repository root:

Use the Python environment at `~/.venvs/myenv` (`source ~/.venvs/myenv/bin/activate`).

- `pip install -r requirements.txt` installs the project dependencies.
- `pytest` runs the public test suite; `pytest -k lup` selects tests by name.
- `ruff check src --select F401,F841` checks for unused imports and variables. Install Ruff separately if it is not available.
- `pyrefly check --preset default --python-interpreter-path ~/.venvs/myenv/bin/python --search-path src src tests` checks types; run it alongside Ruff and pytest.
- `python -m compileall -q src` checks Python syntax, as in CI.
- `docker run --pull always --rm -v "$(pwd)/src:/workspace/src:ro" cr.ct.itmo.ru/numla/hw-checker/equations-basic:latest` runs the public and closed checker tests; it requires Docker and access to the image.

## Coding Style & Naming Conventions

Do not add decorators or code comments. Keep only the original docstrings. Use ordinary conditions for numerical checks.

Follow the neighboring PCA repository's `src/pca.py`: four-space indentation, `numpy.typing.NDArray` annotations for parameters, returns, and local arrays, and `np.floating` for NumPy scalars. Keep mathematical names (`L`, `U`, `d`, `p`); use descriptive names such as `pivot_idx` for indices. Write Russian error messages with diagnostic values, e.g. `{k=}`. Preserve API names, docstrings, shapes, dtypes, and exception contracts. Implement algorithms directly; public tests prohibit library factorizations and solvers such as `numpy.linalg.solve` and `scipy.linalg` in designated implementation files.

## Testing Guidelines

Use pytest and name new files and functions `test_*.py` and `test_*`. Add focused cases for numerical accuracy, edge cases, and array layouts when changing an algorithm. Existing markers include `algorithm`, `cli`, `stress`, and `heavy`; select them with, for example, `pytest -m algorithm`. No coverage threshold is configured. Closed tests run in CI and are not included in the repository.

## Commit & Pull Request Guidelines

The available history contains only an initial commit, so it establishes no detailed commit convention. Use a short, imperative subject describing the change. For a merge request, describe the implementation and testing performed, link any relevant issue, and call out numerical assumptions or remaining limitations. CI runs for merge request events.
