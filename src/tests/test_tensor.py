from pathlib import Path

import numpy as np
from binsparse.container import NPZBinsparseContainer
from binsparse.tensor import (
    BinsparseTensor,
    CSRMatrix,
    CustomTensor,
    DenseLevel,
    DVECVector,
    ISOElementLevel,
    SparseLevel,
)


def test_csr_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "csr.npz"
    original = CSRMatrix(
        (3, 4),
        3,
        fill=True,
        fill_value=np.float32(-1),
        pointers_to_1=np.array([0, 1, 1, 3], dtype=np.uint64),
        indices_1=np.array([0, 1, 3], dtype=np.uint32),
        values=np.array([2, 4, 8], dtype=np.float32),
    )

    with NPZBinsparseContainer(path, "w") as container:
        original.serialize(container)
    with NPZBinsparseContainer(path) as container:
        parsed = BinsparseTensor.parse(container)

    assert isinstance(parsed, CSRMatrix)
    assert parsed.shape == original.shape
    assert parsed.fill_value == original.fill_value
    np.testing.assert_array_equal(parsed.pointers_to_1, original.pointers_to_1)
    np.testing.assert_array_equal(parsed.indices_1, original.indices_1)
    np.testing.assert_array_equal(parsed.values, original.values)


def test_predefined_format_parses_as_custom(tmp_path: Path) -> None:
    path = tmp_path / "csr.npz"
    tensor = CSRMatrix(
        (2, 3),
        1,
        pointers_to_1=np.array([0, 0, 1], dtype=np.uint64),
        indices_1=np.array([2], dtype=np.uint32),
        values=np.array([5], dtype=np.int8),
    )
    with NPZBinsparseContainer(path, "w") as container:
        tensor.serialize(container)
    with NPZBinsparseContainer(path) as container:
        parsed = CustomTensor.parse(container)

    assert isinstance(parsed.level, DenseLevel)
    assert isinstance(parsed.level.level, SparseLevel)


def test_iso_parse_uses_zero_stride_array(tmp_path: Path) -> None:
    path = tmp_path / "custom.npz"
    tensor = CustomTensor(
        (3, 4),
        2,
        level=DenseLevel(
            1,
            SparseLevel(
                1,
                ISOElementLevel(np.int8(7)),
                (np.array([1, 3], dtype=np.uint16),),
                np.array([0, 0, 1, 2], dtype=np.uint64),
            ),
        ),
    )
    with NPZBinsparseContainer(path, "w") as container:
        tensor.serialize(container)
        assert container.read_header()["format"] == "CSR"
    with NPZBinsparseContainer(path) as container:
        parsed = BinsparseTensor.parse(container)
        custom = CustomTensor.parse(container)

    assert isinstance(parsed, CSRMatrix)
    assert parsed.values.shape == (2,)
    assert parsed.values.strides == (0,)
    np.testing.assert_array_equal(parsed.values, [7, 7])
    assert isinstance(custom.level.level.level, ISOElementLevel)
    assert custom.level.level.level.value == 7


def test_complex_buffer_is_decoded_by_container_layer(tmp_path: Path) -> None:
    path = tmp_path / "complex.npz"
    original = DVECVector(
        (2,),
        2,
        values=np.array([1 + 2j, 3 + 4j], dtype=np.complex64),
    )
    with NPZBinsparseContainer(path, "w") as container:
        original.serialize(container)
        assert container.read_header()["data_types"]["values"] == "complex[float32]"
    with NPZBinsparseContainer(path) as container:
        parsed = BinsparseTensor.parse(container)

    assert isinstance(parsed, DVECVector)
    np.testing.assert_array_equal(
        parsed.values, np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    )
