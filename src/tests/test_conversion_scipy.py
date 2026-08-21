"""SciPy conversion tests, skipped when the optional extra is absent."""

import numpy as np
import pytest

from binsparse.conversions import from_scipy, to_scipy
from binsparse.tensor import COORMatrix, CSCMatrix, CSRMatrix

scipy_sparse = pytest.importorskip("scipy.sparse")


@pytest.mark.parametrize(
    ("constructor", "tensor_type"),
    [
        (scipy_sparse.csr_array, CSRMatrix),
        (scipy_sparse.csc_array, CSCMatrix),
        (scipy_sparse.coo_array, COORMatrix),
    ],
)
def test_scipy_round_trip(constructor, tensor_type) -> None:
    source = constructor(np.array([[0, 2, 0], [3, 0, 4]], dtype=np.float32))
    tensor = from_scipy(source)
    result = to_scipy(tensor)

    assert isinstance(tensor, tensor_type)
    assert tensor.shape == source.shape
    np.testing.assert_array_equal(result.toarray(), source.toarray())
    assert result.dtype == source.dtype


def test_scipy_csr_is_sorted_and_unique() -> None:
    source = scipy_sparse.csr_array(
        (
            np.array([1, 2, 3, 4, 5, 6]),
            np.array([2, 1, 2, 1, 0, 1]),
            np.array([0, 3, 6]),
        ),
        shape=(2, 3),
    )
    tensor = from_scipy(source)
    result = to_scipy(tensor)

    assert tensor.number_of_stored_values == 4
    assert result.has_canonical_format
    np.testing.assert_array_equal(result.toarray(), source.toarray())
    assert not source.has_canonical_format


def test_scipy_coo_is_sorted_and_unique() -> None:
    source = scipy_sparse.coo_array(
        (
            np.array([1, 2, 3, 4]),
            (np.array([1, 0, 1, 0]), np.array([0, 2, 0, 2])),
        ),
        shape=(2, 3),
    )
    tensor = from_scipy(source)
    result = to_scipy(tensor)

    assert tensor.number_of_stored_values == 2
    assert result.has_canonical_format
    np.testing.assert_array_equal(result.toarray(), source.toarray())
    assert not source.has_canonical_format


def test_scipy_copy_policy() -> None:
    source = scipy_sparse.csr_array(np.eye(3))

    shared = from_scipy(source, copy=False)
    copied = from_scipy(source, copy=True)

    assert isinstance(shared, CSRMatrix)
    assert isinstance(copied, CSRMatrix)
    assert np.shares_memory(shared.values, source.data)
    assert not np.shares_memory(copied.values, source.data)
    shared_result = to_scipy(shared, copy=False)
    copied_result = to_scipy(shared, copy=True)
    assert np.shares_memory(shared_result.data, shared.values)
    assert not np.shares_memory(copied_result.data, shared.values)


def test_scipy_copy_false_rejects_required_canonicalization() -> None:
    source = scipy_sparse.coo_array(
        (np.array([1, 2]), (np.array([0, 0]), np.array([1, 1]))),
        shape=(2, 2),
    )
    with pytest.raises(ValueError, match="canonicalize"):
        from_scipy(source, copy=False)

    tensor = COORMatrix(
        (2, 2),
        1,
        indices_0=np.array([0]),
        indices_1=np.array([1]),
        values=np.array([3]),
    )
    with pytest.raises(ValueError, match="canonicalize"):
        to_scipy(tensor, copy=False)
