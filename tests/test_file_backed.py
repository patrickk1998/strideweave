import gc
import struct

import pytest
from neotorch import DataType, FileBacked
from neotorch.data.file_backed.data import _session_directory


def test_file_backed_named_file_is_created_in_hidden_session_directory():
    data = FileBacked("named.bin")

    assert data.path.name == "named.bin"
    assert data.path.parent == _session_directory()
    assert data.path.parent.name.startswith(".neotorch-filebacked-")
    assert data.path.exists()


def test_file_backed_random_filename_is_generated_when_omitted():
    first = FileBacked()
    second = FileBacked()

    assert first.path != second.path
    assert first.path.exists()
    assert second.path.exists()


def test_file_backed_rejects_duplicate_filenames():
    existing = FileBacked("duplicate.bin")

    with pytest.raises(FileExistsError):
        FileBacked("duplicate.bin")
    assert existing.path.exists()


def test_file_backed_rejects_filenames_with_path_separators():
    with pytest.raises(ValueError, match="bare file name"):
        FileBacked("nested/name.bin")


def test_file_backed_rejects_unsupported_dtypes():
    with pytest.raises(ValueError, match="dtype"):
        FileBacked(dtype=DataType.Any)

    with pytest.raises(TypeError, match="DataType"):
        FileBacked(dtype="Floating")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (DataType.Floating, [1.5, -2.0, 3.0], [1.5, -2.0, 3.0]),
        (DataType.Float32, [1.5, -2.0, 3.0], [1.5, -2.0, 3.0]),
        (DataType.Int32, [1, -2, 3], [1, -2, 3]),
    ],
)
def test_file_backed_get_set_roundtrip(dtype, values, expected):
    data = FileBacked(dtype=dtype)
    data._allocate(len(values))

    for index, value in enumerate(values):
        data[index] = value

    assert [data[index] for index in range(len(values))] == expected
    assert data.size() == len(values)
    assert data.type() is dtype


def test_file_backed_allocation_zero_fills_storage():
    data = FileBacked()
    data._allocate(4)

    assert data.path.stat().st_size == 4 * struct.calcsize("d")
    assert [data[index] for index in range(4)] == [0.0, 0.0, 0.0, 0.0]


def test_file_backed_int32_set_value_requires_integers():
    data = FileBacked(dtype=DataType.Int32)
    data._allocate(1)

    with pytest.raises(TypeError):
        data[0] = 1.5


def test_file_backed_immutable_rejects_writes():
    data = FileBacked(mutable=False)
    data._allocate(1)

    assert not data.is_mutable()
    with pytest.raises(RuntimeError, match="not mutable"):
        data[0] = 1.0


def test_file_backed_index_out_of_range_raises():
    data = FileBacked()
    data._allocate(2)

    with pytest.raises(IndexError):
        data[2]
    with pytest.raises(IndexError):
        data[-1]


def test_file_backed_new_like_writes_values_and_zeroes_holes():
    data = FileBacked()

    copy = data.new_like([1.0, None, 3.0])

    assert isinstance(copy, FileBacked)
    assert copy.path != data.path
    assert [copy[index] for index in range(3)] == [1.0, 0.0, 3.0]


def test_file_backed_deleting_data_removes_file():
    data = FileBacked()
    path = data.path

    del data
    gc.collect()

    assert not path.exists()


def test_file_backed_release_removes_file_and_blocks_access():
    data = FileBacked()
    data._allocate(1)
    path = data.path

    data.release()

    assert data.is_released()
    assert not path.exists()
    assert data.size() == 0
    with pytest.raises(RuntimeError, match="released"):
        data[0]
    with pytest.raises(RuntimeError, match="released"):
        data.get_value(0)


def test_file_backed_does_not_support_dispatched_operations():
    data = FileBacked()
    for operation_name in ["add", "matmul", "relu", "view", "rearrange"]:
        with pytest.raises(NotImplementedError):
            data.dispatch_op(operation_name)


def test_file_backed_empty_like_allocates_storage_without_values():
    data = FileBacked(dtype=DataType.Int32)

    result = data.empty_like(4, mutable=False)

    assert result.size() == 4
    assert result.type() is DataType.Int32
    assert not result.is_mutable()


def test_file_backed_does_not_support_scatter():
    data = FileBacked()

    with pytest.raises(NotImplementedError):
        data.scatter(None, None, None)
