import gc
import struct

import pytest

from strideweave import DType, FileBacked
from strideweave.carriers.file_backed.carrier import _session_directory


def test_file_backed_named_file_is_created_in_hidden_session_directory():
    carrier = FileBacked("named.bin")

    assert carrier.path.name == "named.bin"
    assert carrier.path.parent == _session_directory()
    assert carrier.path.parent.name.startswith(".strideweave-filebacked-")
    assert carrier.path.exists()


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
        FileBacked(dtype=DType.Any)

    with pytest.raises(TypeError, match="DType"):
        FileBacked(dtype="Floating")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        (DType.Floating, [1.5, -2.0, 3.0], [1.5, -2.0, 3.0]),
        (DType.Float32, [1.5, -2.0, 3.0], [1.5, -2.0, 3.0]),
        (DType.Int32, [1, -2, 3], [1, -2, 3]),
    ],
)
def test_file_backed_get_set_roundtrip(dtype, values, expected):
    carrier = FileBacked(dtype=dtype)
    carrier._allocate(len(values))

    for index, value in enumerate(values):
        carrier[index] = value

    assert [carrier[index] for index in range(len(values))] == expected
    assert carrier.size() == len(values)
    assert carrier.dtype() is dtype


def test_file_backed_allocation_zero_fills_storage():
    carrier = FileBacked()
    carrier._allocate(4)

    assert carrier.path.stat().st_size == 4 * struct.calcsize("d")
    assert [carrier[index] for index in range(4)] == [0.0, 0.0, 0.0, 0.0]


def test_file_backed_int32_set_value_requires_integers():
    carrier = FileBacked(dtype=DType.Int32)
    carrier._allocate(1)

    with pytest.raises(TypeError):
        carrier[0] = 1.5


def test_file_backed_immutable_rejects_writes():
    carrier = FileBacked(mutable=False)
    carrier._allocate(1)

    assert not carrier.is_mutable()
    with pytest.raises(RuntimeError, match="not mutable"):
        carrier[0] = 1.0


def test_file_backed_index_out_of_range_raises():
    carrier = FileBacked()
    carrier._allocate(2)

    with pytest.raises(IndexError):
        carrier[2]
    with pytest.raises(IndexError):
        carrier[-1]


def test_file_backed_new_like_writes_values_and_zeroes_holes():
    carrier = FileBacked()

    copy = carrier.new_like([1.0, None, 3.0])

    assert isinstance(copy, FileBacked)
    assert copy.path != carrier.path
    assert [copy[index] for index in range(3)] == [1.0, 0.0, 3.0]


def test_file_backed_deleting_data_removes_file():
    carrier = FileBacked()
    path = carrier.path

    del carrier
    gc.collect()

    assert not path.exists()


def test_file_backed_release_removes_file_and_blocks_access():
    carrier = FileBacked()
    carrier._allocate(1)
    path = carrier.path

    carrier.release()

    assert carrier.is_released()
    assert not path.exists()
    assert carrier.size() == 0
    with pytest.raises(RuntimeError, match="released"):
        carrier[0]
    with pytest.raises(RuntimeError, match="released"):
        carrier.get_value(0)


def test_file_backed_does_not_support_dispatched_operations():
    carrier = FileBacked()
    for operation_name in ["add", "matmul", "relu", "view", "rearrange"]:
        with pytest.raises(NotImplementedError):
            carrier.dispatch_op(operation_name)


def test_file_backed_allocate_like_allocates_storage_without_values():
    carrier = FileBacked(dtype=DType.Int32)

    result = carrier.allocate_like(4, mutable=False, empty=True)

    assert result.size() == 4
    assert result.dtype() is DType.Int32
    assert not result.is_mutable()


def test_file_backed_does_not_support_scatter():
    carrier = FileBacked()

    with pytest.raises(NotImplementedError):
        carrier.scatter(None, None, None)
