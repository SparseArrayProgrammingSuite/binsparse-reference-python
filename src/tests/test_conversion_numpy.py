"""NumPy conversion tests."""

import numpy as np
import pytest

from binsparse.conversions import from_numpy, to_numpy
from binsparse.tensor import (
    BinsparseTensor,
    CustomTensor,
    DenseLevel,
    DMATRMatrix,
    DVECVector,
)


@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 3, 4)])
def test_numpy_round_trip(shape: tuple[int, ...]) -> None:
    source = np.arange(np.prod(shape)).reshape(shape)
    tensor = from_numpy(source, copy=False)
    result = to_numpy(tensor, copy=False)

    expected_type: type[BinsparseTensor] = (
        DVECVector if len(shape) == 1 else DMATRMatrix
    )
    if len(shape) > 2:
        expected_type = CustomTensor
        assert isinstance(tensor, CustomTensor)
        assert isinstance(tensor.level, DenseLevel)
        assert tensor.level.rank == len(shape)
    assert isinstance(tensor, expected_type)
    assert np.shares_memory(result, source)
    np.testing.assert_array_equal(result, source)


def test_numpy_copy_policy() -> None:
    source = np.arange(6).reshape(2, 3)
    shared = from_numpy(source, copy=False)
    copied = from_numpy(source, copy=True)

    assert isinstance(shared, DMATRMatrix)
    assert isinstance(copied, DMATRMatrix)
    assert np.shares_memory(shared.values, source)
    assert not np.shares_memory(copied.values, source)
    assert np.shares_memory(to_numpy(shared, copy=False), shared.values)
    assert not np.shares_memory(to_numpy(shared, copy=True), shared.values)


def test_numpy_copy_false_rejects_noncontiguous_input() -> None:
    source = np.arange(12).reshape(3, 4).T

    with pytest.raises(ValueError):
        from_numpy(source, copy=False)

    result = to_numpy(from_numpy(source), copy=False)
    np.testing.assert_array_equal(result, source)
