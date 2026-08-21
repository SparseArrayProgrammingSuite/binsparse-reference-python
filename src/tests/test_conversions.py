"""Tests shared by the optional conversion adapters."""

import sys
from collections.abc import Callable
from typing import Any

import pytest

from binsparse.conversions import from_scipy, from_sparse, from_torch


@pytest.mark.parametrize(
    ("dependency", "convert", "extra"),
    [
        ("scipy", from_scipy, "scipy"),
        ("torch", from_torch, "torch"),
        ("sparse", from_sparse, "sparse"),
    ],
)
def test_missing_optional_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    convert: Callable[[Any], Any],
    extra: str,
) -> None:
    monkeypatch.setitem(sys.modules, dependency, None)
    with pytest.raises(ImportError, match=rf"binsparse\[{extra}\]"):
        convert(None)
