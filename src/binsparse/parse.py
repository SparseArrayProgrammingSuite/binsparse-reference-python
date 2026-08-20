"""Container adapters for reading and writing Binsparse data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
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


def _materialize(dataset: Any) -> Any:
    """Read an array from common container APIs while preserving plain values."""
    if not hasattr(dataset, "__getitem__"):
        return dataset
    for selection in ((), Ellipsis):
        try:
            return dataset[selection]
        except (IndexError, TypeError, ValueError):
            continue
    return dataset


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
        return _materialize(self.group[key])

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.group:
            del self.group[key]
        self.group.create_dataset(key, data=value)


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
        return _materialize(self.group[key])

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.group:
            del self.group[key]
        if hasattr(self.group, "create_array"):
            self.group.create_array(key, data=value)
        else:
            self.group.create_dataset(key, data=value)


class NPZBinsparseFile(BinsparseFile):
    """Adapt an NPZ archive or a mutable mapping of NPZ-style entries.

    ``numpy.lib.npyio.NpzFile`` handles are read-only.  To create a new archive,
    populate a mutable mapping through this adapter and pass it to
    ``numpy.savez``.
    """

    def __init__(self, archive: Any):
        self.archive = archive

    @property
    def header(self) -> dict[str, Any]:
        try:
            value = self.archive[BINSPARSE_HEADER]
        except KeyError as error:
            raise KeyError("NPZ archive has no 'binsparse' entry") from error
        return _decode_json(value)

    @header.setter
    def header(self, value: dict[str, Any]) -> None:
        self._require_writable()
        self.archive[BINSPARSE_HEADER] = _encode_json(value)

    def __getitem__(self, key: str) -> Any:
        if key == BINSPARSE_HEADER:
            raise KeyError("use the 'header' property to access binsparse metadata")
        return _materialize(self.archive[key])

    def __setitem__(self, key: str, value: Any) -> None:
        if key == BINSPARSE_HEADER:
            raise KeyError("use the 'header' property to set binsparse metadata")
        self._require_writable()
        self.archive[key] = value

    def _require_writable(self) -> None:
        if not isinstance(self.archive, MutableMapping):
            raise TypeError(
                "this NPZ archive is read-only; use a mutable mapping and "
                "numpy.savez to create a new archive"
            )


__all__ = [
    "BINSPARSE_HEADER",
    "BinsparseFile",
    "HDF5BinsparseFile",
    "ZarrBinsparseFile",
    "NPZBinsparseFile",
]
