"""Container adapters for reading and writing Binsparse data."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from typing import Any


BINSPARSE_HEADER = "binsparse"


def _decode_json(value: Any) -> dict[str, Any]:
    """Decode a JSON object stored as text, bytes, or a scalar array."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError("the binsparse header must be a JSON object")
    return value


def _encode_json(value: dict[str, Any]) -> str:
    if not isinstance(value, dict):
        raise TypeError("the binsparse header must be a dictionary")
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))


class BinsparseFile(ABC):
    """Common interface to a Binsparse binary container or container group."""

    @property
    @abstractmethod
    def header(self) -> dict[str, Any]:
        """Return the decoded Binsparse JSON descriptor."""

    @header.setter
    @abstractmethod
    def header(self, value: dict[str, Any]) -> None:
        """Store a Binsparse JSON descriptor."""

    @abstractmethod
    def __getitem__(self, key: str) -> Any:
        """Read and materialize a named binary array."""

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None:
        """Create or replace a named binary array."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush pending changes without closing the container."""

    @abstractmethod
    def close(self) -> None:
        """Finalize and release the underlying container."""

    def __enter__(self) -> BinsparseFile:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


class HDF5BinsparseFile(BinsparseFile):
    """Adapt an h5py ``File`` or ``Group``."""

    def __init__(self, group: Any):
        self.group = group

    @property
    def header(self) -> dict[str, Any]:
        try:
            value = self.group.attrs[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("HDF5 group has no 'binsparse' attribute") from error
        return _decode_json(value)

    @header.setter
    def header(self, value: dict[str, Any]) -> None:
        self.group.attrs[BINSPARSE_HEADER] = _encode_json(value)

    def __getitem__(self, key: str) -> Any:
        return self.group[key][()]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.group:
            del self.group[key]
        self.group.create_dataset(key, data=value)

    def finalize(self) -> None:
        self.group.file.flush()

    def close(self) -> None:
        self.finalize()
        self.group.file.close()


class ZarrBinsparseFile(BinsparseFile):
    """Adapt a Zarr group without requiring Zarr as a dependency."""

    def __init__(self, group: Any):
        self.group = group

    @property
    def header(self) -> dict[str, Any]:
        try:
            value = self.group.attrs[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("Zarr group has no 'binsparse' attribute") from error
        return _decode_json(value)

    @header.setter
    def header(self, value: dict[str, Any]) -> None:
        _encode_json(value)  # Validate shape and JSON serializability.
        self.group.attrs[BINSPARSE_HEADER] = value

    def __getitem__(self, key: str) -> Any:
        return self.group[key][...]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.group:
            del self.group[key]
        if hasattr(self.group, "create_array"):
            self.group.create_array(key, data=value)
        else:
            self.group.create_dataset(key, data=value)

    def finalize(self) -> None:
        store = getattr(self.group, "store", None)
        flush = getattr(store, "flush", None)
        if flush is not None:
            flush()

    def close(self) -> None:
        self.finalize()
        store = getattr(self.group, "store", None)
        close = getattr(store, "close", None)
        if close is not None:
            close()


class NPZBinsparseFile(BinsparseFile):
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

    @property
    def header(self) -> dict[str, Any]:
        self._require_open()
        try:
            value = self.archive[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("NPZ archive has no 'binsparse' entry") from error
        return _decode_json(value)

    @header.setter
    def header(self, value: dict[str, Any]) -> None:
        self._require_open_and_writable()
        self.archive[BINSPARSE_HEADER] = _encode_json(value)
        self._dirty = True

    def __getitem__(self, key: str) -> Any:
        self._require_open()
        if key == BINSPARSE_HEADER:
            raise KeyError("use the 'header' property to access binsparse metadata")
        return self.archive[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == BINSPARSE_HEADER:
            raise KeyError("use the 'header' property to set binsparse metadata")
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
    "BinsparseFile",
    "HDF5BinsparseFile",
    "ZarrBinsparseFile",
    "NPZBinsparseFile",
]
