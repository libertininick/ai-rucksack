"""General utilities."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter


@contextmanager
def measure_execution_time() -> Generator[Callable[[], float], None, None]:
    """Measure execution time inside with statement.

    Example
    -------
    >>> import time
    >>> with measure_execution_time() as get_time:
    ...     time.sleep(1.)
    >>> print(f"{get_time():.0f}")
    1
    """
    # initialize t1 and t2 with the current timestamp to ensure initial diff = 0
    t1 = t2 = perf_counter()

    # Define a function for calculating the execution time within the context block
    def _calc_execution_time() -> float:
        return t2 - t1

    yield _calc_execution_time

    # Update t2 when the context manager exits
    t2 = perf_counter()


def get_parent_directory_path(
    path: Path | str, parent_directory_name: str
) -> Path | None:
    """Find parent directory by name and return its path.

    Parameters
    ----------
    path: Path | str
        Path to search.
    parent_directory_name: str
        Name of directory to look for.

    Returns
    -------
    Path | None
        Path of parent directory if found, otherwise `None`

    Examples
    --------
    >>> from pathlib import Path
    >>> p = "/some/path/parent1/to/parent2/my-file.txt"

    >>> get_parent_directory_path(p, "parent1")
    PosixPath('/some/path/parent1')

    >>> get_parent_directory_path(p, "parent2")
    PosixPath('/some/path/parent1/to/parent2')

    >>> get_parent_directory_path(p, "parent")

    """
    # Resolve absolute path
    abs_path = Path(path).resolve()

    # Break path into parts
    path_parts = abs_path.parts

    # Find index of parent directory by name in path parts
    try:
        parent_index = path_parts.index(parent_directory_name)
        return Path(*path_parts[: parent_index + 1])
    except ValueError:
        # Parent directory doesn't exist
        return None
