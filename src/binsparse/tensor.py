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
class ISOElementLevel(BinsparseLevel):
    value: Any
    level_desc: ClassVar[str] = "element"
    rank: ClassVar[int] = 0

    @property
    def values(self) -> Any:
        return self.value


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
    buffer_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        self.shape = tuple(self.shape)
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in self.shape
        ):
            raise ValueError("shape dimensions must be non-negative integers")
        if (
            not isinstance(self.number_of_stored_values, int)
            or isinstance(self.number_of_stored_values, bool)
            or self.number_of_stored_values < 0
        ):
            raise ValueError("number_of_stored_values must be a non-negative integer")
        if self.fill is not None and not isinstance(self.fill, bool):
            raise ValueError("fill must be a boolean when present")
        if self.fill is not True and self.fill_value is not None:
            raise ValueError("fill_value requires fill=True")

    @classmethod
    def parse(cls, container: BinsparseContainer) -> BinsparseTensor:
        """Parse a tensor and all required arrays from *container*."""
        header = container.read_header()
        required = ("version", "format", "shape", "number_of_stored_values", "data_types")
        missing = next((key for key in required if key not in header), None)
        if missing is not None:
            raise BinsparseParseError(f"missing required descriptor key {missing!r}")
        if header["version"] != BINSPARSE_VERSION:
            raise BinsparseParseError(
                f"unsupported Binsparse version {header['version']!r}; "
                f"expected {BINSPARSE_VERSION!r}"
            )
        if not isinstance(header["data_types"], dict):
            raise BinsparseParseError("data_types must be an object")
        format_name = header["format"]
        try:
            tensor_cls = (
                CustomTensor
                if format_name == "custom" or cls is CustomTensor
                else _FORMAT_CLASSES[format_name]
            )
        except KeyError as error:
            raise BinsparseParseError(f"unrecognized format {format_name!r}") from error
        if cls is not BinsparseTensor and cls is not tensor_cls:
            raise BinsparseParseError(
                f"descriptor format {format_name!r} cannot be parsed as {cls.__name__}"
            )

        common: dict[str, Any] = {
            "shape": tuple(header["shape"]),
            "number_of_stored_values": header["number_of_stored_values"],
            "fill": header.get("fill"),
        }
        if common["fill"] is True:
            fill = container.read_buffer("fill_value")
            if fill.size != 1:
                raise BinsparseParseError("fill_value must contain exactly one value")
            common["fill_value"] = fill.reshape(-1)[0]
        return tensor_cls._parse(container, header, common)

    @classmethod
    @abstractmethod
    def _parse(
        cls,
        container: BinsparseContainer,
        header: dict[str, Any],
        common: dict[str, Any],
    ) -> BinsparseTensor:
        """Parse format-specific data after shared metadata has been parsed."""

    def serialize(self, container: BinsparseContainer) -> None:
        """Write shared metadata, format-specific data, and the descriptor."""
        data_types: dict[str, str] = {}
        header: dict[str, Any] = {
            "version": BINSPARSE_VERSION,
            "format": self.format,
            "shape": list(self.shape),
            "number_of_stored_values": self.number_of_stored_values,
            "data_types": data_types,
        }
        if self.fill is not None:
            header["fill"] = self.fill
        if self.fill is True:
            fill = np.asarray([self.fill_value])
            data_types["fill_value"] = container.write_buffer("fill_value", fill)
        header.update(self._serialize(container, data_types))
        container.write_header(header)

    @abstractmethod
    def _serialize(
        self,
        container: BinsparseContainer,
        data_types: dict[str, str],
    ) -> dict[str, Any]:
        """Write format-specific arrays and return additional descriptor data."""


class _PredefinedTensor(BinsparseTensor):
    @classmethod
    def _parse(
        cls,
        container: BinsparseContainer,
        header: dict[str, Any],
        common: dict[str, Any],
    ) -> BinsparseTensor:
        buffers: dict[str, Any] = {
            name: container.read_buffer(name) for name in cls.buffer_names
        }
        return cls(**common, **buffers)

    def _serialize(
        self,
        container: BinsparseContainer,
        data_types: dict[str, str],
    ) -> dict[str, Any]:
        for name in self.buffer_names:
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"required buffer {name!r} is missing")
            data_types[name] = container.write_buffer(name, np.asarray(value))
        return {}


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
        if isinstance(level, (ElementLevel, ISOElementLevel)):
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
        level = cls._parse_level(container, header, descriptor, 0)
        return cls(
            **common,
            level=level,
            transpose=None if transpose is None else tuple(transpose),
        )

    @staticmethod
    def _parse_level(
        container: BinsparseContainer,
        header: dict[str, Any],
        descriptor: dict[str, Any],
        depth: int,
    ) -> BinsparseLevel:
        try:
            level_desc = descriptor["level_desc"]
        except (KeyError, TypeError) as error:
            raise BinsparseParseError("each custom level requires level_desc") from error
        if level_desc == "element":
            values = container.read_buffer("values")
            array = np.asarray(values)
            if array.ndim == 1 and array.strides == (0,):
                return ISOElementLevel(array[0])
            return ElementLevel(values)
        if level_desc not in {"dense", "sparse"}:
            raise BinsparseParseError(f"unrecognized level descriptor {level_desc!r}")
        try:
            rank = descriptor["rank"]
            child_descriptor = descriptor["level"]
        except KeyError as error:
            raise BinsparseParseError(f"{level_desc} level requires {error.args[0]!r}") from error
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise BinsparseParseError(f"{level_desc} level rank must be >= 1")
        child = CustomTensor._parse_level(container, header, child_descriptor, depth + rank)
        if level_desc == "dense":
            return DenseLevel(rank, child)
        pointers = None if depth == 0 else container.read_buffer(f"pointers_to_{depth}")
        indices = tuple(
            container.read_buffer(f"indices_{dimension}")
            for dimension in range(depth, depth + rank)
        )
        return SparseLevel(rank, child, indices, pointers)

    def _serialize(
        self,
        container: BinsparseContainer,
        data_types: dict[str, str],
    ) -> dict[str, Any]:
        descriptor = self._serialize_level(container, self.level, 0, data_types)
        custom: dict[str, Any] = {"level": descriptor}
        if self.transpose is not None:
            custom["transpose"] = list(self.transpose)
        result: dict[str, Any] = {"custom": custom}
        for alias, (alias_descriptor, alias_transpose) in _PREDEFINED_LEVELS.items():
            if descriptor == alias_descriptor and self.transpose == alias_transpose:
                result["format"] = alias
                break
        return result

    @staticmethod
    def _serialize_level(
        container: BinsparseContainer,
        level: BinsparseLevel | None,
        depth: int,
        data_types: dict[str, str],
    ) -> dict[str, Any]:
        if isinstance(level, ISOElementLevel):
            value = np.asarray([level.value])
            values = np.broadcast_to(value, (1,))
            data_types["values"] = container.write_buffer("values", values)
            return {"level_desc": "element"}
        if isinstance(level, ElementLevel):
            values = np.asarray(level.values)
            data_types["values"] = container.write_buffer("values", values)
            return {"level_desc": "element"}
        if isinstance(level, DenseLevel):
            child = CustomTensor._serialize_level(
                container, level.level, depth + level.rank, data_types
            )
            return {"level_desc": "dense", "rank": level.rank, "level": child}
        if isinstance(level, SparseLevel):
            if depth > 0:
                if level.pointers_to_next is None:
                    raise ValueError(f"sparse level at depth {depth} requires pointers")
                name = f"pointers_to_{depth}"
                pointers = np.asarray(level.pointers_to_next)
                data_types[name] = container.write_buffer(name, pointers)
            for offset, indices in enumerate(level.indices):
                name = f"indices_{depth + offset}"
                data_types[name] = container.write_buffer(name, np.asarray(indices))
            child = CustomTensor._serialize_level(
                container, level.level, depth + level.rank, data_types
            )
            return {"level_desc": "sparse", "rank": level.rank, "level": child}
        raise TypeError(f"unsupported level type {type(level).__name__}")


@dataclass
class DVECVector(_PredefinedTensor):
    values: Any = None
    format: ClassVar[str] = "DVEC"
    buffer_names: ClassVar[tuple[str, ...]] = ("values",)


@dataclass
class DMATRMatrix(_PredefinedTensor):
    values: Any = None
    format: ClassVar[str] = "DMATR"
    buffer_names: ClassVar[tuple[str, ...]] = ("values",)


@dataclass
class DMATCMatrix(_PredefinedTensor):
    values: Any = None
    format: ClassVar[str] = "DMATC"
    buffer_names: ClassVar[tuple[str, ...]] = ("values",)


DMATMatrix = DMATRMatrix


@dataclass
class CVECVector(_PredefinedTensor):
    indices_0: Any = None
    values: Any = None
    format: ClassVar[str] = "CVEC"
    buffer_names: ClassVar[tuple[str, ...]] = ("indices_0", "values")


@dataclass
class CSRMatrix(_PredefinedTensor):
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "CSR"
    buffer_names: ClassVar[tuple[str, ...]] = ("pointers_to_1", "indices_1", "values")


@dataclass
class CSCMatrix(CSRMatrix):
    format: ClassVar[str] = "CSC"


@dataclass
class DCSRMatrix(_PredefinedTensor):
    indices_0: Any = None
    pointers_to_1: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "DCSR"
    buffer_names: ClassVar[tuple[str, ...]] = (
        "indices_0", "pointers_to_1", "indices_1", "values"
    )


@dataclass
class DCSCMatrix(DCSRMatrix):
    format: ClassVar[str] = "DCSC"


@dataclass
class COORMatrix(_PredefinedTensor):
    indices_0: Any = None
    indices_1: Any = None
    values: Any = None
    format: ClassVar[str] = "COOR"
    buffer_names: ClassVar[tuple[str, ...]] = ("indices_0", "indices_1", "values")


@dataclass
class COOCMatrix(COORMatrix):
    format: ClassVar[str] = "COOC"


COOMatrix = COORMatrix


_FORMAT_CLASSES: dict[str, type[BinsparseTensor]] = {
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


__all__ = [
    "BinsparseLevel", "IndexableLevel", "ElementLevel", "ISOElementLevel",
    "DenseLevel", "SparseLevel", "BinsparseTensor", "CustomTensor",
    "DVECVector", "DMATRMatrix", "DMATCMatrix", "DMATMatrix", "CVECVector",
    "CSRMatrix", "CSCMatrix", "DCSRMatrix", "DCSCMatrix", "COORMatrix",
    "COOCMatrix", "COOMatrix",
]
