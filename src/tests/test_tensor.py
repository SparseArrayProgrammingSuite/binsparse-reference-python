import numpy as np
import pytest
from binsparse.container import NPZBinsparseContainer
from binsparse.errors import BinsparseParseError
from binsparse.tensor import (
    BinsparseTensor,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DVECVector,
    ElementLevel,
    SparseLevel,
)


def test_csr_round_trip() -> None:
    archive: dict[str, np.ndarray] = {}
    original = CSRMatrix(
        (3, 4),
        3,
        fill=True,
        fill_value=np.float32(-1),
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([0, 1, 3], dtype=np.uint32),
        values=np.array([2, 4, 8], dtype=np.float32),
    )

    container = NPZBinsparseContainer(archive)
    original.serialize(container)
    parsed = BinsparseTensor.parse(container)

    assert isinstance(parsed, CSRMatrix)
    assert parsed.shape == original.shape
    assert parsed.fill_value == original.fill_value
    np.testing.assert_array_equal(parsed.pointers_to_1, original.pointers_to_1)
    np.testing.assert_array_equal(parsed.indices_1, original.indices_1)
    np.testing.assert_array_equal(parsed.values, original.values)


def test_explicit_format_requires_exact_match() -> None:
    archive: dict[str, np.ndarray] = {}
    tensor = CSRMatrix(
        (2, 3),
        1,
        pointers_to_1=np.array([0, 0, 1], dtype=np.uint64),
        indices_1=np.array([2], dtype=np.uint32),
        values=np.array([5], dtype=np.int8),
    )
    container = NPZBinsparseContainer(archive)
    tensor.serialize(container, alias=True)
    with pytest.raises(BinsparseParseError):
        CustomTensor.parse(container)


def test_iso_parse_uses_zero_stride_array() -> None:
    archive: dict[str, np.ndarray] = {}
    tensor = CustomTensor(
        (3, 4),
        2,
        level=DenseLevel(
            1,
            SparseLevel(
                1,
                ElementLevel(
                    np.broadcast_to(np.array([7], dtype=np.int8), (2,))
                ),
                (np.array([1, 3], dtype=np.uint16),),
                np.array([0, 0, 1, 2], dtype=np.uint64),
            ),
        ),
    )
    container = NPZBinsparseContainer(archive)
    tensor.serialize(container, alias=True)
    assert container.read_header()["format"] == "CSR"
    parsed = BinsparseTensor.parse(container)

    assert isinstance(parsed, CSRMatrix)
    assert parsed.values.shape == (2,)
    assert parsed.values.strides == (0,)
    np.testing.assert_array_equal(parsed.values, [7, 7])
    with pytest.raises(BinsparseParseError):
        CustomTensor.parse(container)


def test_complex_buffer_is_decoded_by_container_layer() -> None:
    archive: dict[str, np.ndarray] = {}
    original = DVECVector(
        (2,),
        2,
        values=np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
    )
    container = NPZBinsparseContainer(archive)
    original.serialize(container)
    assert container.read_header()["data_types"]["values"] == "complex[float32]"
    parsed = BinsparseTensor.parse(container)

    assert isinstance(parsed, DVECVector)
    np.testing.assert_array_equal(
        parsed.values, np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    )


def test_nested_iso_complex_buffer_round_trip() -> None:
    archive: dict[str, np.ndarray] = {}
    values = np.broadcast_to(np.array([1 + 2j], dtype=np.complex64), (3,))
    original = DVECVector((3,), 3, values=values)

    container = NPZBinsparseContainer(archive)
    original.serialize(container)
    assert container.read_header()["data_types"]["values"] == (
        "iso[complex[float32]]"
    )
    parsed = BinsparseTensor.parse(container)

    assert isinstance(parsed, DVECVector)
    assert parsed.values.strides == (0,)
    np.testing.assert_array_equal(parsed.values, values)
    with pytest.raises(BinsparseParseError, match="expected_size is required"):
        container.read_buffer("values")


def test_alias_controls_serialized_and_parsed_representation() -> None:
    original = CSRMatrix(
        (2, 3),
        1,
        pointers_to_1=np.array([0, 0, 1]),
        indices_1=np.array([2]),
        values=np.array([5]),
    )
    predefined_archive: dict[str, np.ndarray] = {}
    predefined_container = NPZBinsparseContainer(predefined_archive)
    original.serialize(predefined_container)
    assert predefined_container.read_header()["format"] == "CSR"
    assert isinstance(
        BinsparseTensor.parse(predefined_container, alias=False), CustomTensor
    )

    archive: dict[str, np.ndarray] = {}
    container = NPZBinsparseContainer(archive)
    original.serialize(container, alias=False)

    assert container.read_header()["format"] == "custom"
    assert isinstance(BinsparseTensor.parse(container), CustomTensor)
    assert isinstance(BinsparseTensor.parse(container, alias=False), CustomTensor)
    assert isinstance(BinsparseTensor.parse(container, alias=True), CSRMatrix)

    custom = BinsparseTensor.parse(container, alias=False)
    alias_archive: dict[str, np.ndarray] = {}
    alias_container = NPZBinsparseContainer(alias_archive)
    custom.serialize(alias_container, alias=True)
    assert alias_container.read_header()["format"] == "CSR"


def test_tensor_copy_policy_uses_one_container_boundary_copy() -> None:
    values = np.arange(4, dtype=np.float32)
    tensor = DVECVector((4,), 4, values=values)

    shared_archive: dict[str, np.ndarray] = {}
    shared_container = NPZBinsparseContainer(shared_archive)
    tensor.serialize(shared_container, copy=False)
    assert np.shares_memory(shared_archive["values"], values)
    shared = BinsparseTensor.parse(shared_container, copy=False)
    assert isinstance(shared, DVECVector)
    assert np.shares_memory(shared.values, shared_archive["values"])

    copied_archive: dict[str, np.ndarray] = {}
    copied_container = NPZBinsparseContainer(copied_archive)
    tensor.serialize(copied_container, copy=True)
    assert not np.shares_memory(copied_archive["values"], values)
    copied = BinsparseTensor.parse(copied_container, copy=True)
    assert isinstance(copied, DVECVector)
    assert not np.shares_memory(copied.values, copied_archive["values"])
