"""Conversions between Binsparse tensors and PyData/Sparse arrays."""

from typing import Any

import numpy as np

from binsparse.tensor import (
    BinsparseTensor,
    COORMatrix,
    CustomTensor,
    ElementLevel,
    SparseLevel,
)


def _sparse() -> Any:
    try:
        import sparse
    except ImportError as error:
        raise ImportError(
            "PyData/Sparse conversions require the 'sparse' extra: "
            "pip install binsparse[sparse]"
        ) from error
    return sparse


def from_sparse(value: Any, *, copy: bool | None = None) -> BinsparseTensor:
    """Convert an N-dimensional PyData/Sparse COO array to Binsparse."""
    sparse = _sparse()
    if not isinstance(value, sparse.COO) or value.ndim < 1:
        raise TypeError("expected a non-scalar PyData/Sparse COO array")
    if copy is True:
        value = value.copy(deep=True)
    fill_value = np.asarray(value.fill_value).item()
    shape = tuple(value.shape)
    count = int(value.data.size)
    indices = tuple(value.coords[dimension, :] for dimension in range(value.ndim))
    values = value.data
    if value.ndim == 2:
        return COORMatrix(
            shape,
            count,
            fill=True,
            fill_value=fill_value,
            indices_0=indices[0],
            indices_1=indices[1],
            values=values,
        )
    return CustomTensor(
        shape,
        count,
        fill=True,
        fill_value=fill_value,
        level=SparseLevel(value.ndim, ElementLevel(values), indices),
    )


def to_sparse(tensor: BinsparseTensor, *, copy: bool | None = None) -> Any:
    """Convert a flat Binsparse COO tensor to a PyData/Sparse COO array."""
    sparse = _sparse()
    if isinstance(tensor, COORMatrix):
        indices = (tensor.indices_0, tensor.indices_1)
        values = tensor.values
    elif (
        isinstance(tensor, CustomTensor)
        and tensor.transpose is None
        and isinstance(tensor.level, SparseLevel)
        and tensor.level.rank == len(tensor.shape)
        and tensor.level.pointers_to_next is None
        and isinstance(tensor.level.level, ElementLevel)
    ):
        indices = tensor.level.indices
        values = tensor.level.level.values
    else:
        raise TypeError(f"cannot convert {type(tensor).__name__} to PyData/Sparse")
    if copy is False:
        raise ValueError(
            "copy=False requires Binsparse to support storing COO indices in a "
            "single coordinate matrix"
        )
    coords = np.stack(indices)
    fill_value = tensor.fill_value if tensor.fill is True else 0
    return sparse.COO(
        coords,
        np.array(values, copy=True) if copy is True else values,
        shape=tensor.shape,
        fill_value=fill_value,
    )


__all__ = ["from_sparse", "to_sparse"]
