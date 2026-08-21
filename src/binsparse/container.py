"""Container adapters for reading and writing Binsparse data."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any

import numpy as np

from .dtypes import dtype_to_str, str_to_dtype
from .errors import BinsparseParseError


BINSPARSE_HEADER = "binsparse"


class BinsparseContainer(ABC):
    """Common interface to a Binsparse binary container or container group."""

    def read_header(self) -> dict[str, Any]:
        """Return the decoded Binsparse JSON descriptor."""
        return self._read_header()

    @abstractmethod
    def _read_header(self) -> dict[str, Any]:
        """Read and decode the Binsparse header from the backend."""

    def write_header(self, value: dict[str, Any]) -> None:
        """Store a Binsparse JSON descriptor."""
        if not isinstance(value, dict):
            raise TypeError("the binsparse header must be a dictionary")
        json.dumps(value)
        self._write_header(value)

    @abstractmethod
    def _write_header(self, value: dict[str, Any]) -> None:
        """Write a validated Binsparse header to the backend."""

    def read_buffer(self, key: str) -> np.ndarray:
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
                decoded = decode(match.group(1), data)
                if decoded.size != 1:
                    raise BinsparseParseError("an ISO buffer must contain one value")
                return np.broadcast_to(
                    decoded.reshape(1), (header["number_of_stored_values"],)
                )
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
        return decode(declared, self._read_buffer(key))

    @abstractmethod
    def _read_buffer(self, key: str) -> np.ndarray:
        """Read a named binary array without applying Binsparse decoding."""

    def write_buffer(
        self,
        key: str,
        value: np.ndarray,
    ) -> str:
        """Encode, create or replace a named array, returning its data type."""
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

        encoded, data_type = encode(np.asarray(value))
        self._write_buffer(key, encoded)
        return data_type

    @abstractmethod
    def _write_buffer(self, key: str, value: np.ndarray) -> None:
        """Write an already encoded named binary array."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush pending changes without closing the container."""

    @abstractmethod
    def close(self) -> None:
        """Finalize and release the underlying container."""

    def __enter__(self) -> BinsparseContainer:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class HDF5BinsparseContainer(BinsparseContainer):
    """Adapt an h5py ``Container`` or ``Group``."""

    def __init__(self, group: Any):
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

    def finalize(self) -> None:
        pass

    def close(self) -> None:
        pass


class ZarrBinsparseContainer(BinsparseContainer):
    """Adapt a Zarr group without requiring Zarr as a dependency."""

    def __init__(self, group: Any):
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

    def finalize(self) -> None:
        pass

    def close(self) -> None:
        pass


class NPZBinsparseContainer(BinsparseContainer):
    """Read or assemble an NPZ archive and persist it on finalization.

    ``mode`` follows the familiar file conventions: ``"r"`` reads an existing
    archive, ``"w"`` creates one, and ``"a"`` loads an existing archive for
    modification (or creates it if it does not exist).
    """

    def __init__(self, file: Any, mode: str = "r"):
        if mode not in {"r", "w", "a"}:
            raise ValueError("NPZ mode must be 'r', 'w', or 'a'")

        import numpy

        self.file = file
        self.mode = mode
        self._dirty = False
        self._closed = False

        if mode == "w":
            self.archive: Any = {}
        else:
            try:
                loaded = numpy.load(file, allow_pickle=False)
            except FileNotFoundError:
                if mode == "r":
                    raise
                self.archive = {}
            else:
                if mode == "r":
                    self.archive = loaded
                else:
                    self.archive = {key: loaded[key] for key in loaded.files}
                    loaded.close()

    def _read_header(self) -> dict[str, Any]:
        self._require_open()
        try:
            value = self.archive[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("NPZ archive has no 'binsparse' entry") from error
        return json.loads(str(value.item()))

    def _write_header(self, value: dict[str, Any]) -> None:
        self._require_open_and_writable()
        self.archive[BINSPARSE_HEADER] = np.asarray(
            json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))
        )
        self._dirty = True

    def _read_buffer(self, key: str) -> np.ndarray:
        self._require_open()
        if key == BINSPARSE_HEADER:
            raise KeyError("use read_header() to access binsparse metadata")
        return np.asarray(self.archive[key])

    def _write_buffer(self, key: str, value: np.ndarray) -> None:
        if key == BINSPARSE_HEADER:
            raise KeyError("use write_header() to set binsparse metadata")
        self._require_open_and_writable()
        self.archive[key] = value
        self._dirty = True

    def finalize(self) -> None:
        self._require_open()
        if self.mode != "r" and self._dirty:
            import numpy

            numpy.savez(self.file, **self.archive)
            self._dirty = False

    def close(self) -> None:
        if self._closed:
            return
        self.finalize()
        close = getattr(self.archive, "close", None)
        if close is not None:
            close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("operation on closed NPZ Binsparse file")

    def _require_open_and_writable(self) -> None:
        self._require_open()
        if self.mode == "r":
            raise TypeError("NPZ Binsparse file is not writable")


__all__ = [
    "BINSPARSE_HEADER",
    "BinsparseContainer",
    "HDF5BinsparseContainer",
    "ZarrBinsparseContainer",
    "NPZBinsparseContainer",
]
