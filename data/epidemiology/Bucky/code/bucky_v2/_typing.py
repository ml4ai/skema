"""Provide some generic types for mypy."""
import os
from typing import Union

from numpy.typing import ArrayLike as np_ArrayLike

PathLike = Union[str, os.PathLike]
ArrayLike = np_ArrayLike
