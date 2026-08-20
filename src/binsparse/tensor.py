"""In-memory representations of the Binsparse tensor formats.

Attribute names mirror the descriptor and binary-array names in the spec.
Array-like objects are typed as ``Any`` so callers may use any backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .version import BINSPARSE_VERSION


class BinsparseLevel:
    """Base class for a level in a custom Binsparse format."""

    level_desc: ClassVar[str]


@dataclass
class IndexableLevel(BinsparseLevel):
    """Base class for levels which consume tensor dimensions."""

    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("an indexable level must have rank >= 1")


@dataclass
class ElementLevel(BinsparseLevel):
    """A collection of scalar values (a rank-zero terminal level)."""

    values: Any
    level_desc: ClassVar[str] = "element"
    rank: ClassVar[int] = 0


@dataclass
class ISOElementLevel(BinsparseLevel):
    """An element level whose logical values are all the same scalar."""

    value: Any
    level_desc: ClassVar[str] = "element"
    rank: ClassVar[int] = 0

    @property
    def values(self) -> Any:
        return self.value


@dataclass
class DenseLevel(IndexableLevel):
    """One or more row-major dense dimensions followed by a sublevel."""

    level: BinsparseLevel
    level_desc: ClassVar[str] = "dense"


@dataclass
class SparseLevel(IndexableLevel):
    """One or more sparse dimensions followed by a sublevel."""

    level: BinsparseLevel
    indices: tuple[Any, ...]
    pointers_to_next: Any | None = None
    level_desc: ClassVar[str] = "sparse"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.indices) != self.rank:
            raise ValueError("the number of indices arrays must equal rank")


@dataclass
class BinsparseTensor:
    """Metadata shared by every Binsparse tensor representation."""

    shape: tuple[int, ...]
    number_of_stored_values: int
    fill: bool | None = None
    fill_value: Any = None

    version: ClassVar[str] = BINSPARSE_VERSION
    format: ClassVar[str]

    def __post_init__(self) -> None:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if self.number_of_stored_values < 0:
            raise ValueError("number_of_stored_values must be non-negative")
        if self.fill is not True and self.fill_value is not None:
            raise ValueError("fill_value requires fill=True")


@dataclass
class CustomTensor(BinsparseTensor):
    level: BinsparseLevel | None = None
    transpose: tuple[int, ...] | None = None
    format: ClassVar[str] = "custom"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.level is None:
            raise ValueError("a custom tensor requires a root level")
        if self.transpose is not None:
            dimensions = set(range(len(self.shape)))
            if len(self.transpose) != len(self.shape) or set(self.transpose) != dimensions:
                raise ValueError("transpose must be a permutation of the dimensions")


@dataclass
class DVECVector(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DVEC"


@dataclass
class DMATRMatrix(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DMATR"


@dataclass
class DMATCMatrix(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DMATC"


DMATMatrix = DMATRMatrix


@dataclass
class CVECVector(BinsparseTensor):
    indices_0: Any = None
    values: Any = None
    format: ClassVar[str] = "CVEC"


@dataclass
class CSRMatrix(BinsparseTensor):
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "CSR"


@dataclass
class CSCMatrix(BinsparseTensor):
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "CSC"


@dataclass
class DCSRMatrix(BinsparseTensor):
    indices_0: Any = None
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "DCSR"


@dataclass
class DCSCMatrix(BinsparseTensor):
    indices_0: Any = None
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "DCSC"


@dataclass
class COORMatrix(BinsparseTensor):
    indices_0: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "COOR"


@dataclass
class COOCMatrix(BinsparseTensor):
    indices_0: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "COOC"


COOMatrix = COORMatrix


__all__ = [
    "BinsparseLevel", "IndexableLevel", "ElementLevel", "ISOElementLevel",
    "DenseLevel", "SparseLevel", "BinsparseTensor", "CustomTensor",
    "DVECVector", "DMATRMatrix", "DMATCMatrix", "DMATMatrix", "CVECVector",
    "CSRMatrix", "CSCMatrix", "DCSRMatrix", "DCSCMatrix", "COORMatrix",
    "COOCMatrix", "COOMatrix",
]
