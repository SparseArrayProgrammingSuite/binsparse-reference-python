"""PyTorch conversion tests, skipped when the optional extra is absent."""

import numpy as np
import pytest

from binsparse.conversions import from_torch, to_torch
from binsparse.tensor import (
    COORMatrix,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DMATRMatrix,
    DVECVector,
    ElementLevel,
    SparseLevel,
)

torch = pytest.importorskip("torch")


def test_torch_dense_round_trip() -> None:
    source = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    tensor = from_torch(source)
    result = to_torch(tensor)

    assert isinstance(tensor, DMATRMatrix)
    np.testing.assert_array_equal(result.numpy(), source.numpy())


def test_torch_nd_dense_round_trip() -> None:
    source = torch.arange(24).reshape(2, 3, 4)
    tensor = from_torch(source, copy=False)
    result = to_torch(tensor, copy=False)

    assert isinstance(tensor, CustomTensor)
    assert isinstance(tensor.level, DenseLevel)
    assert tensor.level.rank == source.ndim
    assert result.data_ptr() == source.data_ptr()
    assert torch.equal(result, source)


def test_torch_csr_round_trip() -> None:
    source = torch.tensor([[0.0, 2.0], [3.0, 0.0]]).to_sparse_csr()
    tensor = from_torch(source)
    result = to_torch(tensor)

    assert isinstance(tensor, CSRMatrix)
    assert torch.equal(result.to_dense(), source.to_dense())


def test_torch_uncoalesced_coo_preserves_entries() -> None:
    source = torch.sparse_coo_tensor(
        torch.tensor([[0, 0], [1, 1]]),
        torch.tensor([2.0, 3.0]),
        (2, 2),
    )
    tensor = from_torch(source)

    assert isinstance(tensor, COORMatrix)
    assert tensor.number_of_stored_values == 2


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_torch_nd_coo_round_trip(shape) -> None:
    coordinates = torch.tensor([[0, size - 1] for size in shape])
    source = torch.sparse_coo_tensor(coordinates, torch.tensor([2.0, 3.0]), shape)
    tensor = from_torch(source, copy=False)
    result = to_torch(tensor)

    assert isinstance(tensor, CustomTensor)
    assert isinstance(tensor.level, SparseLevel)
    assert tensor.level.rank == len(shape)
    assert isinstance(tensor.level.level, ElementLevel)
    assert all(
        index.__array_interface__["data"][0]
        == source._indices()[dimension].data_ptr()
        for dimension, index in enumerate(tensor.level.indices)
    )
    assert torch.equal(result.to_dense(), source.to_dense())
    with pytest.raises(ValueError, match="combine"):
        to_torch(tensor, copy=False)


def test_torch_copy_policy() -> None:
    source = torch.tensor([1.0, 2.0])
    shared = from_torch(source, copy=False)
    copied = from_torch(source, copy=True)

    assert isinstance(shared, DVECVector)
    assert isinstance(copied, DVECVector)
    assert shared.values.__array_interface__["data"][0] == source.data_ptr()
    assert copied.values.__array_interface__["data"][0] != source.data_ptr()

    shared_result = to_torch(shared, copy=False)
    copied_result = to_torch(shared, copy=True)
    assert shared_result.data_ptr() == shared.values.__array_interface__["data"][0]
    assert copied_result.data_ptr() != shared.values.__array_interface__["data"][0]


def test_torch_copy_true_handles_tensors_requiring_grad() -> None:
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    copied = from_torch(source, copy=True)

    assert isinstance(copied, DVECVector)
    assert copied.values.__array_interface__["data"][0] != source.data_ptr()
    np.testing.assert_array_equal(copied.values, source.detach().numpy())
