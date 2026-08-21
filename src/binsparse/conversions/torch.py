"""Conversions between Binsparse tensors and PyTorch tensors."""

from typing import Any

import numpy as np

from binsparse.conversions.numpy import from_numpy, to_numpy
from binsparse.tensor import (
    BinsparseTensor,
    COORMatrix,
    CSCMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATCMatrix,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
    SparseLevel,
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "PyTorch conversions require the 'torch' extra: "
            "pip install binsparse[torch]"
        ) from error
    return torch


def _numpy(value: Any, copy: bool | None) -> np.ndarray:
    if copy is True:
        try:
            return value.numpy(force=False).copy()
        except (RuntimeError, TypeError):
            result = value.numpy(force=True)
            shares_storage = (
                value.device.type == "cpu"
                and result.__array_interface__["data"][0] == value.data_ptr()
            )
            return result.copy() if shares_storage else result
    if copy is False:
        return value.numpy(force=False)
    try:
        return value.numpy(force=False)
    except (RuntimeError, TypeError):
        return value.numpy(force=True)


def from_torch(value: Any, *, copy: bool | None = None) -> BinsparseTensor:
    """Convert a PyTorch dense, COO, CSR, or CSC tensor to Binsparse."""
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("expected a PyTorch tensor")

    if value.layout == torch.strided:
        array = _numpy(value, copy)
        return from_numpy(array, copy=False if copy is True else copy)

    if value.layout == torch.sparse_coo:
        indices = _numpy(value._indices(), copy)
        values = _numpy(value._values(), copy)
        if value.dense_dim() != 0:
            raise TypeError("hybrid sparse COO tensors are not supported")
        if value.sparse_dim() == 2:
            return COORMatrix(
                tuple(value.shape),
                int(values.size),
                indices_0=indices[0, :],
                indices_1=indices[1, :],
                values=values,
            )
        return CustomTensor(
            tuple(value.shape),
            int(values.size),
            level=SparseLevel(
                value.sparse_dim(),
                ElementLevel(values),
                tuple(
                    indices[dimension, :]
                    for dimension in range(value.sparse_dim())
                ),
            ),
        )

    if value.layout == torch.sparse_csr:
        values = _numpy(value.values(), copy)
        return CSRMatrix(
            tuple(value.shape),
            int(values.size),
            pointers_to_1=_numpy(value.crow_indices(), copy),
            indices_1=_numpy(value.col_indices(), copy),
            values=values,
        )
    if value.layout == torch.sparse_csc:
        values = _numpy(value.values(), copy)
        return CSCMatrix(
            tuple(value.shape),
            int(values.size),
            pointers_to_1=_numpy(value.ccol_indices(), copy),
            indices_1=_numpy(value.row_indices(), copy),
            values=values,
        )
    raise TypeError(f"unsupported PyTorch layout {value.layout}")


def _tensor(torch: Any, value: Any, device: Any, copy: bool | None) -> Any:
    return torch.asarray(value, device=device, copy=copy)


def to_torch(
    tensor: BinsparseTensor,
    *,
    device: Any = None,
    copy: bool | None = None,
) -> Any:
    """Convert a supported Binsparse tensor to PyTorch, optionally on *device*."""
    torch = _torch()
    if (
        isinstance(tensor, (DVECVector, DMATRMatrix, DMATCMatrix))
        or isinstance(tensor, CustomTensor)
        and tensor.transpose is None
        and (
            isinstance(tensor.level, ElementLevel)
            or isinstance(tensor.level, DenseLevel)
            and tensor.level.rank == len(tensor.shape)
            and isinstance(tensor.level.level, ElementLevel)
        )
    ):
        return _tensor(torch, to_numpy(tensor, copy=False), device, copy)

    if tensor.fill is True and tensor.fill_value != 0:
        raise ValueError("PyTorch conversion requires a zero fill value")
    if isinstance(tensor, CSCMatrix):
        return torch.sparse_csc_tensor(
            _tensor(torch, tensor.pointers_to_1, device, copy),
            _tensor(torch, tensor.indices_1, device, copy),
            _tensor(torch, tensor.values, device, copy),
            size=tensor.shape,
            device=device,
        )
    if isinstance(tensor, CSRMatrix):
        return torch.sparse_csr_tensor(
            _tensor(torch, tensor.pointers_to_1, device, copy),
            _tensor(torch, tensor.indices_1, device, copy),
            _tensor(torch, tensor.values, device, copy),
            size=tensor.shape,
            device=device,
        )
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
        raise TypeError(f"cannot convert {type(tensor).__name__} to PyTorch")
    if copy is False:
        raise ValueError("copy=False cannot combine Binsparse COO coordinate arrays")
    return torch.sparse_coo_tensor(
        _tensor(torch, np.stack(indices), device, copy),
        _tensor(torch, values, device, copy),
        size=tensor.shape,
        device=device,
    )


__all__ = ["from_torch", "to_torch"]
