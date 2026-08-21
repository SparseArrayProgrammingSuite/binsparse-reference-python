"""Conversions between Binsparse tensors and SciPy sparse arrays."""

from typing import Any

from binsparse.tensor import BinsparseTensor, COORMatrix, CSCMatrix, CSRMatrix


def _scipy_sparse() -> Any:
    try:
        import scipy.sparse as scipy_sparse
    except ImportError as error:
        raise ImportError(
            "SciPy conversions require the 'scipy' extra: pip install binsparse[scipy]"
        ) from error
    return scipy_sparse


def _prepare(value: Any, copy: bool | None) -> Any:
    if value.has_canonical_format:
        return value.copy() if copy is True else value
    if copy is False:
        raise ValueError("copy=False cannot canonicalize a SciPy sparse array")
    result = value.copy()
    result.sum_duplicates()
    return result


def from_scipy(value: Any, *, copy: bool | None = None) -> BinsparseTensor:
    """Convert a two-dimensional SciPy CSR, CSC, or COO object to Binsparse."""
    scipy_sparse = _scipy_sparse()
    if not scipy_sparse.issparse(value) or value.ndim != 2:
        raise TypeError("expected a two-dimensional SciPy sparse array or matrix")
    if value.format not in {"coo", "csr", "csc"}:
        raise TypeError(f"unsupported SciPy sparse format {value.format!r}")
    value = _prepare(value, copy)

    shape = tuple(value.shape)
    count = int(value.data.size)
    values = value.data
    if value.format == "csr":
        return CSRMatrix(
            shape,
            count,
            pointers_to_1=value.indptr,
            indices_1=value.indices,
            values=values,
        )
    if value.format == "csc":
        return CSCMatrix(
            shape,
            count,
            pointers_to_1=value.indptr,
            indices_1=value.indices,
            values=values,
        )
    if value.format == "coo":
        return COORMatrix(
            shape,
            count,
            indices_0=value.row,
            indices_1=value.col,
            values=values,
        )
    raise TypeError(f"unsupported SciPy sparse format {value.format!r}")


def to_scipy(tensor: BinsparseTensor, *, copy: bool | None = None) -> Any:
    """Convert a Binsparse CSR, CSC, or COO matrix to a SciPy sparse array."""
    scipy_sparse = _scipy_sparse()
    if tensor.fill is True and tensor.fill_value != 0:
        raise ValueError("SciPy conversion requires a zero fill value")
    if isinstance(tensor, CSCMatrix):
        result = scipy_sparse.csc_array(
            (tensor.values, tensor.indices_1, tensor.pointers_to_1),
            shape=tensor.shape,
            copy=copy,
        )
    elif isinstance(tensor, CSRMatrix):
        result = scipy_sparse.csr_array(
            (tensor.values, tensor.indices_1, tensor.pointers_to_1),
            shape=tensor.shape,
            copy=copy,
        )
    elif isinstance(tensor, COORMatrix):
        result = scipy_sparse.coo_array(
            (tensor.values, (tensor.indices_0, tensor.indices_1)),
            shape=tensor.shape,
            copy=copy,
        )
    else:
        raise TypeError(f"cannot convert {type(tensor).__name__} to SciPy")
    return result


__all__ = ["from_scipy", "to_scipy"]
