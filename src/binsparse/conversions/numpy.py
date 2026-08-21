"""Conversions between Binsparse tensors and NumPy arrays."""

from typing import Literal

import numpy as np

from binsparse.tensor import (
    BinsparseTensor,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
)


def from_numpy(value: np.ndarray, *, copy: bool | None = None) -> BinsparseTensor:
    """Convert a NumPy array to a dense Binsparse tensor."""
    if not isinstance(value, np.ndarray):
        raise TypeError("expected a NumPy ndarray")
    array = np.array(value, copy=copy, order="C")
    values = array.reshape(-1)
    common = (tuple(array.shape), int(array.size))
    if array.ndim == 1:
        return DVECVector(*common, values=values)
    if array.ndim == 2:
        return DMATRMatrix(*common, values=values)
    level = ElementLevel(values)
    return CustomTensor(
        *common,
        level=level if array.ndim == 0 else DenseLevel(array.ndim, level),
    )


def to_numpy(tensor: BinsparseTensor, *, copy: bool | None = None) -> np.ndarray:
    """Convert a dense Binsparse tensor to a NumPy array."""
    order: Literal["C", "F"] = "C"
    if isinstance(tensor, (DVECVector, DMATRMatrix)):
        values = tensor.values
    elif isinstance(tensor, DMATCMatrix):
        values = tensor.values
        order = "F"
    elif (
        isinstance(tensor, CustomTensor)
        and tensor.transpose is None
        and isinstance(tensor.level, ElementLevel)
        and not tensor.shape
    ):
        values = tensor.level.values
    elif (
        isinstance(tensor, CustomTensor)
        and tensor.transpose is None
        and isinstance(tensor.level, DenseLevel)
        and tensor.level.rank == len(tensor.shape)
        and isinstance(tensor.level.level, ElementLevel)
    ):
        values = tensor.level.level.values
    else:
        raise TypeError(f"cannot convert {type(tensor).__name__} to NumPy")
    return np.array(values, copy=copy).reshape(tensor.shape, order=order)


__all__ = ["from_numpy", "to_numpy"]
