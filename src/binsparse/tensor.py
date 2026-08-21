"""In-memory representations of the Binsparse tensor formats.

Attribute names mirror the descriptor and binary-array names in the spec.
Array-like objects are typed as ``Any`` so callers may use any backend.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar
from .container import BinsparseContainer

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
class BinsparseTensor(ABC):
    """Metadata shared by every Binsparse tensor representation."""

    shape: tuple[int, ...]
    number_of_stored_values: int
    fill: bool | None = None
    fill_value: Any = None
    data_type: Any = None
    format: str
    version: ClassVar[str] = BINSPARSE_VERSION

    def __post_init__(self) -> None:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if self.number_of_stored_values < 0:
            raise ValueError("number_of_stored_values must be non-negative")
        if self.fill is not True and self.fill_value is not None:
            raise ValueError("fill_value requires fill=True")

    @classmethod
    def parse(self, f:BinsparseContainer) -> BinsparseTensor:
        header = f.read_header()
        self.shape = tuple(header["shape"])
        self.number_of_stored_values = header["number_of_stored_values"]
        self.fill = header["fill"]
        if self.fill:
            self.fill_value = f.read_buffer("fill_value")[0]
        if header["version"] != BINSPARSE_VERSION:
            raise BinsparseParseError("Wrong Binsparse Version")

    def serialize(self, f:BinsparseContainer):
        if self.fill:
            f.write_buffer("fill_value", np.full((1,), self.fill_value))
        f.write_header({
            "version": BINSPARSE_VERSION,
            "shape": self.shape
        })


bspread_tensor_lookup = OrderedDict(
    "DVEC" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "element"
            ),
        ),
    ),
    "DMAT" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "dense",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "DMATR" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "dense",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "DMATC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_desc" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "dense",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "CVEC" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "sparse",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "element"
            ),
        ),
    ),
    "CSR" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "CSC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_desc" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "DCSR" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "sparse",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "DCSC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_desc" => "sparse",
            "rank" => 1,
            "level" => OrderedDict(
                "level_desc" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_desc" => "element"
                ),
            ),
        ),
    ),
    "COO" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "sparse",
            "rank" => 2,
            "level" => OrderedDict(
                "level_desc" => "element"
            ),
        ),
    ),
    "COOR" => OrderedDict(
        "level" => OrderedDict(
            "level_desc" => "sparse",
            "rank" => 2,
            "level" => OrderedDict(
                "level_desc" => "element"
            ),
        ),
    ),
    "COOC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_desc" => "sparse",
            "rank" => 2,
            "level" => OrderedDict(
                "level_desc" => "element"
            ),
        ),
    ),
)

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

    def parse_level(self, f, fmt, depth):
        match fmt["level_desc"]:
            case "dense":
                rank = fmt["rank"]
                level = self.parse_level(f, fmt["level"], depth + rank)
                return DenseLevel(rank, level)
            case "sparse":
                rank = fmt["rank"]
                if depth == 0:
                    pointers_to_next = None
                else:
                    pointers_to_next = f.read_buffer(fmt[f"pointers_to_{depth}"])
                indices = [f.read_buffer(fmt[f"indices_{depth + r}"]) for r in range(rank)]
                level = self.parse_level(f, fmt["level"], depth + rank)
                return SparseLevel(rank, level, indices, pointers_to_next)
            case "element":
                values = f.read_buffer("values")
                return ElementLevel(values)
            case desc:
                raise BinsparseParseError(f"unrecognized level descriptor \"{desc}\"")

    def serialize_level(self, f, level, depth):
        match level:
            case DenseLevel(rank, level):
                format = self.serialize_level(f, level, depth)
                return {
                    "level_desc": "dense",
                    "rank": rank,
                    "level": format
                }
            case 
                



    @classmethod
    def parse(self, f:BinsparseContainer):
        super(self).parse()
        header = f.get_header()
        if header["format"] == "custom":
            BinsparseLevel.parse_level(f, header["custom"], 0)
        else:
            BinsparseLevel.parse_level(f, bspread_tensor_lookup[header["format"]], 0)
        
        




@dataclass
class DVECVector(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DVEC"

    def parse(self, f):
        super.parse()
        self.values = f.read_buffer("values")

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
