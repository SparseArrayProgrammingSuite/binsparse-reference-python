"""Container adapters for reading and writing Binsparse data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
import json
import re
from typing import Any

import numpy as np

from .errors import BinsparseParseError


BINSPARSE_HEADER = "binsparse"


dtype_to_str = {
    np.dtype("int8"): "int8",
    np.dtype("int16"): "int16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("uint32"): "uint32",
    np.dtype("uint64"): "uint64",
    np.dtype("float32"): "float32",
    np.dtype("float64"): "float64",
    np.dtype("bool"): "bint8",
}

str_to_dtype = {value: key for key, value in dtype_to_str.items()}

class BinsparseContainer(ABC):
    """Common interface to a Binsparse binary container or container group."""

    def __init__(self) -> None:
        self.data_types: dict[str, str] = {}

    def read_header(self) -> dict[str, Any]:
        """Return the decoded Binsparse JSON descriptor."""
        header = self._read_header()
        if not isinstance(header.get("data_types"), dict):
            raise BinsparseParseError("data_types must be an object")
        return header

    @abstractmethod
    def _read_header(self) -> dict[str, Any]:
        """Read and decode the Binsparse header from the backend."""

    def write_header(self, value: dict[str, Any]) -> None:
        """Store a Binsparse JSON descriptor."""
        if not isinstance(value, dict):
            raise TypeError("the binsparse header must be a dictionary")
        header = {**value, "data_types": self.data_types.copy()}
        json.dumps(header)
        self._write_header(header)

    @abstractmethod
    def _write_header(self, value: dict[str, Any]) -> None:
        """Write a validated Binsparse header to the backend."""

    def read_buffer(
        self,
        key: str,
        expected_size: int | None = None,
        *,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Read and decode a named binary array."""
        header = self.read_header()
        try:
            declared = header["data_types"][key]
        except KeyError as error:
            raise BinsparseParseError(f"missing data type for buffer {key!r}") from error
        if not isinstance(declared, str):
            raise BinsparseParseError(f"data type for buffer {key!r} must be a string")
        def decode(data_type: str, data: np.ndarray) -> np.ndarray:
            if (match := re.fullmatch(r"iso\[(.*)\]", data_type)) is not None:
                if expected_size is None:
                    raise BinsparseParseError(
                        "expected_size is required when reading an ISO buffer"
                    )
                decoded = decode(match.group(1), data)
                if decoded.size != 1:
                    raise BinsparseParseError("an ISO buffer must contain one value")
                return np.broadcast_to(decoded.reshape(1), (expected_size,))
            if (match := re.fullmatch(r"complex\[(.*)\]", data_type)) is not None:
                decoded = decode(match.group(1), data)
                if decoded.dtype not in {np.dtype("float32"), np.dtype("float64")}:
                    raise BinsparseParseError("complex values require float32 or float64")
                if decoded.ndim != 1 or decoded.size % 2 != 0:
                    raise BinsparseParseError("invalid complex value buffer")
                complex_dtype = (
                    np.complex64 if decoded.dtype == np.float32 else np.complex128
                )
                return np.ascontiguousarray(decoded).view(complex_dtype)
            if re.fullmatch(r"[^\[\]]+", data_type) is None:
                raise BinsparseParseError(f"unknown Binsparse type wrapper {data_type!r}")
            try:
                dtype = str_to_dtype[data_type]
            except KeyError as error:
                raise BinsparseParseError(f"unknown Binsparse type {data_type!r}") from error
            return np.asarray(data, dtype=dtype)  # type: ignore[call-overload]
        encoded = self._read_buffer(key)
        data = decode(declared, encoded)
        if expected_size is not None and data.size != expected_size:
            raise BinsparseParseError(
                f"buffer {key!r} has size {data.size}, expected {expected_size}"
            )
        shares_memory = np.shares_memory(data, encoded)
        if copy is False and not shares_memory:
            raise ValueError(f"copy=False cannot decode buffer {key!r} without copying")
        if copy is True and shares_memory:
            return data.copy()
        return data

    @abstractmethod
    def _read_buffer(self, key: str) -> np.ndarray:
        """Read a named binary array without applying Binsparse decoding."""

    def write_buffer(
        self,
        key: str,
        value: np.ndarray,
        *,
        copy: bool | None = None,
    ) -> None:
        """Encode, create or replace a named array and record its data type."""
        def encode(data: np.ndarray) -> tuple[np.ndarray, str]:
            if data.ndim == 1 and data.strides == (0,):
                if data.size == 0:
                    raise ValueError("cannot encode an empty ISO buffer")
                encoded, data_type = encode(data[:1].copy())
                return encoded, f"iso[{data_type}]"
            if np.issubdtype(data.dtype, np.complexfloating):
                if data.dtype not in {np.dtype("complex64"), np.dtype("complex128")}:
                    raise TypeError(f"unsupported complex dtype: {data.dtype}")
                dtype = np.dtype("float32" if data.dtype == np.complex64 else "float64")
                encoded, data_type = encode(np.ascontiguousarray(data).view(dtype))
                return encoded, f"complex[{data_type}]"
            try:
                return data, dtype_to_str[data.dtype]
            except KeyError as error:
                raise TypeError(f"unsupported Binsparse dtype: {data.dtype}") from error

        source = np.asarray(value)
        encoded, data_type = encode(source)
        shares_memory = np.shares_memory(encoded, source)
        if copy is False and not shares_memory and not data_type.startswith("iso["):
            raise ValueError(f"copy=False cannot encode buffer {key!r} without copying")
        if copy is True and shares_memory:
            encoded = encoded.copy()
        self._write_buffer(key, encoded)
        self.data_types[key] = data_type

    @abstractmethod
    def _write_buffer(self, key: str, value: np.ndarray) -> None:
        """Write an already encoded named binary array."""

class HDF5BinsparseContainer(BinsparseContainer):
    """Adapt an h5py ``Container`` or ``Group``."""

    def __init__(self, group: Any):
        super().__init__()
        self.group = group

    def _read_header(self) -> dict[str, Any]:
        try:
            value = self.group.attrs[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("HDF5 group has no 'binsparse' attribute") from error
        return json.loads(value)

    def _write_header(self, value: dict[str, Any]) -> None:
        self.group.attrs[BINSPARSE_HEADER] = json.dumps(
            value, indent=2, sort_keys=True, separators=(",", ": ")
        )

    def _read_buffer(self, key: str) -> np.ndarray:
        return np.asarray(self.group[key][()])

    def _write_buffer(self, key: str, value: np.ndarray) -> None:
        if key in self.group:
            del self.group[key]
        self.group.create_dataset(key, data=value)

class ZarrBinsparseContainer(BinsparseContainer):
    """Adapt a Zarr group without requiring Zarr as a dependency."""

    def __init__(self, group: Any):
        super().__init__()
        self.group = group

    def _read_header(self) -> dict[str, Any]:
        try:
            return self.group.attrs[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("Zarr group has no 'binsparse' attribute") from error

    def _write_header(self, value: dict[str, Any]) -> None:
        self.group.attrs[BINSPARSE_HEADER] = value

    def _read_buffer(self, key: str) -> np.ndarray:
        return np.asarray(self.group[key][...])

    def _write_buffer(self, key: str, value: np.ndarray) -> None:
        if key in self.group:
            del self.group[key]
        if hasattr(self.group, "create_array"):
            self.group.create_array(key, data=value)
        else:
            self.group.create_dataset(key, data=value)

class NPZBinsparseContainer(BinsparseContainer):
    """Adapt a mutable mapping of NPZ entry names to NumPy arrays."""

    def __init__(self, file: MutableMapping[str, np.ndarray]):
        super().__init__()
        self.file = file

    def _read_header(self) -> dict[str, Any]:
        try:
            value = self.file[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("NPZ archive has no 'binsparse' entry") from error
        return json.loads(str(value.item()))

    def _write_header(self, value: dict[str, Any]) -> None:
        self.file[BINSPARSE_HEADER] = np.asarray(
            json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))
        )

    def _read_buffer(self, key: str) -> np.ndarray:
        if key == BINSPARSE_HEADER:
            raise KeyError("use read_header() to access binsparse metadata")
        return self.file[key]

    def _write_buffer(self, key: str, value: np.ndarray) -> None:
        if key == BINSPARSE_HEADER:
            raise KeyError("use write_header() to set binsparse metadata")
        self.file[key] = value

__all__ = [
    "BINSPARSE_HEADER",
    "BinsparseContainer",
    "HDF5BinsparseContainer",
    "ZarrBinsparseContainer",
    "NPZBinsparseContainer",
]
