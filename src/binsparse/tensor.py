"""In-memory representations and container parsing for Binsparse tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from .container import BinsparseContainer
from .errors import BinsparseParseError
from .version import BINSPARSE_VERSION


class BinsparseLevel:
    level_desc: ClassVar[str]


@dataclass
class IndexableLevel(BinsparseLevel):
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("an indexable level must have rank >= 1")


@dataclass
class ElementLevel(BinsparseLevel):
    values: Any
    level_desc: ClassVar[str] = "element"
    rank: ClassVar[int] = 0


@dataclass
class DenseLevel(IndexableLevel):
    level: BinsparseLevel
    level_desc: ClassVar[str] = "dense"


@dataclass
class SparseLevel(IndexableLevel):
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
    shape: tuple[int, ...]
    number_of_stored_values: int
    fill: bool | None = None
    fill_value: Any = None

    version: ClassVar[str] = BINSPARSE_VERSION
    format: ClassVar[str]

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)
        if any(size < 0 for size in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if self.number_of_stored_values < 0:
            raise ValueError("number_of_stored_values must be non-negative")
        if self.fill is not True and self.fill_value is not None:
            raise ValueError("fill_value requires fill=True")

    @classmethod
    def parse(
        cls,
        container: BinsparseContainer,
        *,
        alias: bool | None = None,
        copy: bool | None = None,
    ) -> BinsparseTensor:
        """Parse a tensor and all required arrays from *container*."""
        header = container.read_header()
        required = ("version", "format", "shape", "number_of_stored_values")
        missing = next((key for key in required if key not in header), None)
        if missing is not None:
            raise BinsparseParseError(f"missing required descriptor key {missing!r}")
        if header["version"] != BINSPARSE_VERSION:
            raise BinsparseParseError(
                f"unsupported Binsparse version {header['version']!r}; "
                f"expected {BINSPARSE_VERSION!r}"
            )
        format_name = header["format"]
        try:
            if hasattr(cls, "format"):
                if cls.format != format_name:
                    raise BinsparseParseError(
                        f"descriptor format {format_name!r} cannot be parsed as "
                        f"{cls.__name__}"
                    )
                tensor_cls = cls
            elif alias is False:
                tensor_cls = CustomTensor
            elif alias is True and format_name == "custom":
                alias_name = _custom_alias(header)
                tensor_cls = (
                    CustomTensor
                    if alias_name is None
                    else _FORMAT_CLASSES[alias_name]
                )
            else:
                tensor_cls = _FORMAT_CLASSES[format_name]
        except KeyError as error:
            raise BinsparseParseError(f"unrecognized format {format_name!r}") from error
        common: dict[str, Any] = {
            "shape": tuple(header["shape"]),
            "number_of_stored_values": header["number_of_stored_values"],
            "fill": header.get("fill"),
        }
        if common["fill"] is True:
            fill = container.read_buffer("fill_value", 1, copy=copy)
            if fill.size != 1:
                raise BinsparseParseError("fill_value must contain exactly one value")
            common["fill_value"] = fill.reshape(-1)[0]
        return tensor_cls._parse(container, header, common, copy)

    @classmethod
    @abstractmethod
    def _parse(
        cls,
        container: BinsparseContainer,
        header: dict[str, Any],
        common: dict[str, Any],
        copy: bool | None,
    ) -> BinsparseTensor:
        """Parse format-specific data after shared metadata has been parsed."""

    def serialize(
        self,
        container: BinsparseContainer,
        *,
        alias: bool | None = None,
        copy: bool | None = None,
    ) -> None:
        """Write shared metadata, format-specific data, and the descriptor."""
        header: dict[str, Any] = {
            "version": BINSPARSE_VERSION,
            "format": self.format,
            "shape": list(self.shape),
            "number_of_stored_values": self.number_of_stored_values,
        }
        if self.fill is not None:
            header["fill"] = self.fill
        if self.fill is True:
            fill = np.asarray([self.fill_value])
            container.write_buffer("fill_value", fill, copy=copy)
        header.update(self._serialize(container, copy))
        if alias is False and self.format != "custom":
            descriptor, transpose = _PREDEFINED_LEVELS[self.format]
            custom: dict[str, Any] = {"level": descriptor}
            if transpose is not None:
                custom["transpose"] = list(transpose)
            header.update(format="custom", custom=custom)
        elif alias is True and self.format == "custom":
            alias_name = _custom_alias(header)
            if alias_name is not None:
                header["format"] = alias_name
                header.pop("custom")
        container.write_header(header)

    @abstractmethod
    def _serialize(
        self,
        container: BinsparseContainer,
        copy: bool | None,
    ) -> dict[str, Any]:
        """Write format-specific arrays and return additional descriptor data."""


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
        if self._level_rank(self.level) != len(self.shape):
            raise ValueError("custom levels must consume exactly the tensor rank")

    @staticmethod
    def _level_rank(level: BinsparseLevel) -> int:
        if isinstance(level, ElementLevel):
            return 0
        if isinstance(level, (DenseLevel, SparseLevel)):
            return level.rank + CustomTensor._level_rank(level.level)
        raise TypeError(f"unsupported level type {type(level).__name__}")

    @classmethod
    def _parse(
        cls,
        container: BinsparseContainer,
        header: dict[str, Any],
        common: dict[str, Any],
        copy: bool | None,
    ) -> BinsparseTensor:
        format_name = header["format"]
        if format_name == "custom":
            try:
                custom = header["custom"]
                descriptor = custom["level"]
            except (KeyError, TypeError) as error:
                raise BinsparseParseError("custom format requires custom.level") from error
            transpose = custom.get("transpose")
        else:
            try:
                descriptor, transpose = _PREDEFINED_LEVELS[format_name]
            except KeyError as error:
                raise BinsparseParseError(f"unrecognized format {format_name!r}") from error
        transpose_tuple = None if transpose is None else tuple(transpose)
        level = cls._parse_level(container, header, descriptor, 0, copy)
        return cls(
            **common,
            level=level,
            transpose=transpose_tuple,
        )

    @staticmethod
    def _parse_level(
        container: BinsparseContainer,
        header: dict[str, Any],
        descriptor: dict[str, Any],
        depth: int,
        copy: bool | None,
    ) -> BinsparseLevel:
        try:
            level_desc = descriptor["level_desc"]
        except (KeyError, TypeError) as error:
            raise BinsparseParseError("each custom level requires level_desc") from error
        match level_desc:
            case "element":
                count = header["number_of_stored_values"]
                return ElementLevel(container.read_buffer("values", count, copy=copy))
            case "dense":
                try:
                    rank = descriptor["rank"]
                    child_descriptor = descriptor["level"]
                except KeyError as error:
                    raise BinsparseParseError(
                        f"dense level requires {error.args[0]!r}"
                    ) from error
                if rank < 1:
                    raise BinsparseParseError("dense level rank must be >= 1")
                child = CustomTensor._parse_level(
                    container,
                    header,
                    child_descriptor,
                    depth + rank,
                    copy,
                )
                return DenseLevel(rank, child)
            case "sparse":
                try:
                    rank = descriptor["rank"]
                    child_descriptor = descriptor["level"]
                except KeyError as error:
                    raise BinsparseParseError(
                        f"sparse level requires {error.args[0]!r}"
                    ) from error
                if rank < 1:
                    raise BinsparseParseError("sparse level rank must be >= 1")
                pointers = (
                    None
                    if depth == 0
                    else container.read_buffer(f"pointers_to_{depth}", copy=copy)
                )
                indices = tuple(
                    container.read_buffer(f"indices_{dimension}", copy=copy)
                    for dimension in range(depth, depth + rank)
                )
                child = CustomTensor._parse_level(
                    container,
                    header,
                    child_descriptor,
                    depth + rank,
                    copy,
                )
                return SparseLevel(rank, child, indices, pointers)
            case _:
                raise BinsparseParseError(
                    f"unrecognized level descriptor {level_desc!r}"
                )

    def _serialize(
        self,
        container: BinsparseContainer,
        copy: bool | None,
    ) -> dict[str, Any]:
        descriptor = self._serialize_level(container, self.level, 0, copy)
        custom: dict[str, Any] = {"level": descriptor}
        if self.transpose is not None:
            custom["transpose"] = list(self.transpose)
        return {"custom": custom}

    @staticmethod
    def _serialize_level(
        container: BinsparseContainer,
        level: BinsparseLevel | None,
        depth: int,
        copy: bool | None,
    ) -> dict[str, Any]:
        match level:
            case ElementLevel(values):
                container.write_buffer("values", np.asarray(values), copy=copy)
                return {"level_desc": "element"}
            case DenseLevel(rank, child_level):
                child = CustomTensor._serialize_level(
                    container, child_level, depth + rank, copy
                )
                return {"level_desc": "dense", "rank": rank, "level": child}
            case SparseLevel(rank, child_level, indices, pointers):
                if depth > 0:
                    if pointers is None:
                        raise ValueError(
                            f"sparse level at depth {depth} requires pointers"
                        )
                    name = f"pointers_to_{depth}"
                    container.write_buffer(name, np.asarray(pointers), copy=copy)
                for offset, index in enumerate(indices):
                    name = f"indices_{depth + offset}"
                    container.write_buffer(name, np.asarray(index), copy=copy)
                child = CustomTensor._serialize_level(
                    container, child_level, depth + rank, copy
                )
                return {"level_desc": "sparse", "rank": rank, "level": child}
            case _:
                raise TypeError(f"unsupported level type {type(level).__name__}")


@dataclass
class DVECVector(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DVEC"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        return cls(**common, values=container.read_buffer("values", count, copy=copy))

    def _serialize(self, container, copy):
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


@dataclass
class DMATRMatrix(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DMATR"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        return cls(**common, values=container.read_buffer("values", count, copy=copy))

    def _serialize(self, container, copy):
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


@dataclass
class DMATCMatrix(BinsparseTensor):
    values: Any = None
    format: ClassVar[str] = "DMATC"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        return cls(**common, values=container.read_buffer("values", count, copy=copy))

    def _serialize(self, container, copy):
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


DMATMatrix = DMATRMatrix


@dataclass
class CVECVector(BinsparseTensor):
    indices_0: Any = None
    values: Any = None
    format: ClassVar[str] = "CVEC"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        return cls(
            **common,
            indices_0=container.read_buffer("indices_0", copy=copy),
            values=container.read_buffer("values", count, copy=copy),
        )

    def _serialize(self, container, copy):
        container.write_buffer("indices_0", np.asarray(self.indices_0), copy=copy)
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


@dataclass
class CSRMatrix(BinsparseTensor):
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "CSR"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        return cls(
            **common,
            pointers_to_1=container.read_buffer("pointers_to_1", copy=copy),
            indices_1=container.read_buffer("indices_1", copy=copy),
            values=container.read_buffer("values", count, copy=copy),
        )

    def _serialize(self, container, copy):
        container.write_buffer(
            "pointers_to_1", np.asarray(self.pointers_to_1), copy=copy
        )
        container.write_buffer("indices_1", np.asarray(self.indices_1), copy=copy)
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


@dataclass
class CSCMatrix(CSRMatrix):
    format: ClassVar[str] = "CSC"


@dataclass
class DCSRMatrix(BinsparseTensor):
    indices_0: Any = None
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "DCSR"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        indices_0 = container.read_buffer("indices_0", copy=copy)
        return cls(
            **common,
            indices_0=indices_0,
            pointers_to_1=container.read_buffer("pointers_to_1", copy=copy),
            indices_1=container.read_buffer("indices_1", copy=copy),
            values=container.read_buffer("values", count, copy=copy),
        )

    def _serialize(self, container, copy):
        container.write_buffer("indices_0", np.asarray(self.indices_0), copy=copy)
        container.write_buffer(
            "pointers_to_1", np.asarray(self.pointers_to_1), copy=copy
        )
        container.write_buffer("indices_1", np.asarray(self.indices_1), copy=copy)
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


@dataclass
class DCSCMatrix(DCSRMatrix):
    format: ClassVar[str] = "DCSC"


@dataclass
class COORMatrix(BinsparseTensor):
    indices_0: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "COOR"

    @classmethod
    def _parse(cls, container, header, common, copy):
        count = common["number_of_stored_values"]
        return cls(
            **common,
            indices_0=container.read_buffer("indices_0", copy=copy),
            indices_1=container.read_buffer("indices_1", copy=copy),
            values=container.read_buffer("values", count, copy=copy),
        )

    def _serialize(self, container, copy):
        container.write_buffer("indices_0", np.asarray(self.indices_0), copy=copy)
        container.write_buffer("indices_1", np.asarray(self.indices_1), copy=copy)
        container.write_buffer("values", np.asarray(self.values), copy=copy)
        return {}


@dataclass
class COOCMatrix(COORMatrix):
    format: ClassVar[str] = "COOC"


COOMatrix = COORMatrix


_FORMAT_CLASSES: dict[str, type[BinsparseTensor]] = {
    "custom": CustomTensor,
    "DVEC": DVECVector,
    "DMAT": DMATRMatrix,
    "DMATR": DMATRMatrix,
    "DMATC": DMATCMatrix,
    "CVEC": CVECVector,
    "CSR": CSRMatrix,
    "CSC": CSCMatrix,
    "DCSR": DCSRMatrix,
    "DCSC": DCSCMatrix,
    "COO": COORMatrix,
    "COOR": COORMatrix,
    "COOC": COOCMatrix,
}


_ELEMENT = {"level_desc": "element"}


def _dense(level: dict[str, Any], rank: int = 1) -> dict[str, Any]:
    return {"level_desc": "dense", "rank": rank, "level": level}


def _sparse(level: dict[str, Any], rank: int = 1) -> dict[str, Any]:
    return {"level_desc": "sparse", "rank": rank, "level": level}


_PREDEFINED_LEVELS: dict[str, tuple[dict[str, Any], tuple[int, ...] | None]] = {
    "DVEC": (_dense(_ELEMENT), None),
    "DMATR": (_dense(_dense(_ELEMENT)), None),
    "DMAT": (_dense(_dense(_ELEMENT)), None),
    "DMATC": (_dense(_dense(_ELEMENT)), (1, 0)),
    "CVEC": (_sparse(_ELEMENT), None),
    "CSR": (_dense(_sparse(_ELEMENT)), None),
    "CSC": (_dense(_sparse(_ELEMENT)), (1, 0)),
    "DCSR": (_sparse(_sparse(_ELEMENT)), None),
    "DCSC": (_sparse(_sparse(_ELEMENT)), (1, 0)),
    "COOR": (_sparse(_ELEMENT, 2), None),
    "COO": (_sparse(_ELEMENT, 2), None),
    "COOC": (_sparse(_ELEMENT, 2), (1, 0)),
}


def _custom_alias(header: dict[str, Any]) -> str | None:
    try:
        custom = header["custom"]
        descriptor = custom["level"]
        transpose = custom.get("transpose")
    except (KeyError, TypeError) as error:
        raise BinsparseParseError("custom format requires custom.level") from error
    transpose_tuple = None if transpose is None else tuple(transpose)
    return next(
        (
            alias
            for alias, (alias_descriptor, alias_transpose) in _PREDEFINED_LEVELS.items()
            if descriptor == alias_descriptor and transpose_tuple == alias_transpose
        ),
        None,
    )


__all__ = [
    "BinsparseLevel", "IndexableLevel", "ElementLevel",
    "DenseLevel", "SparseLevel", "BinsparseTensor", "CustomTensor",
    "DVECVector", "DMATRMatrix", "DMATCMatrix", "DMATMatrix", "CVECVector",
    "CSRMatrix", "CSCMatrix", "DCSRMatrix", "DCSCMatrix", "COORMatrix",
    "COOCMatrix", "COOMatrix",
]
