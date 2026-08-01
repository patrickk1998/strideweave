"""Focused checks for the public functional frontend."""

from __future__ import annotations

import pytest

import strideweave as sw


def _tensor(values: list[float], shape: sw.Shape) -> sw.Tensor:
    return sw.Tensor(
        sw.Generic(values, dtype=sw.DType.Float32),
        0,
        sw.Layout(shape, sw.Stride([1, 2, 2])),
    )


def test_conv_general_frontend_normalizes_noncanonical_roles_and_output() -> None:
    # Source modes are [spatial, batch, feature], while the primitive consumes
    # [batch, feature, spatial].  The output requests [spatial, batch, feature].
    lhs = _tensor([1.0, 2.0], sw.Shape([2, 1, 1]))
    kernel = _tensor([2.0], sw.Shape([1, 1, 1]))

    result = sw.conv_general(
        lhs,
        kernel,
        (1,),
        ((0, 0),),
        lhs_dims=(1, 2, 0),
        kernel_dims=(2, 1, 0),
        output_dims=(2, 0, 1),
    )

    assert result.layout.shape == sw.Shape([2, 1, 1])
    assert [result[index] for index in range(result.size())] == [2.0, 4.0]


def test_reduce_sum_is_the_only_top_level_reduction_name() -> None:
    assert not hasattr(sw, "reduce")
    assert hasattr(sw, "reduce_sum")


def test_select_and_clamp_are_public_and_require_matching_carrier_classes() -> None:
    layout = sw.Layout(sw.Shape(1), sw.Stride(1))
    condition = sw.Tensor(sw.Generic([True], dtype=sw.DType.Bool), 0, layout)
    generic = sw.Tensor(sw.Generic([1.0], dtype=sw.DType.Float32), 0, layout)
    cpu = sw.Tensor(sw.CPU(1, dtype=sw.DType.Float32), 0, layout)

    assert "select" in sw.__all__
    assert "clamp" in sw.__all__
    assert sw.GenericSelectOperation is not None
    assert sw.GenericClampOperation is not None
    with pytest.raises(TypeError, match="carriers must match"):
        sw.select(condition, generic, cpu)
    with pytest.raises(TypeError, match="carriers must match"):
        sw.clamp(generic, cpu, 2.0)


def test_select_and_clamp_delegate_exact_generic_semantics() -> None:
    layout = sw.Layout(sw.Shape(3), sw.Stride(1))
    condition = sw.Tensor(
        sw.Generic([True, False, True], dtype=sw.DType.Bool), 0, layout
    )
    on_true = sw.Tensor(sw.Generic([1.0, 2.0, 3.0], dtype=sw.DType.Float32), 0, layout)
    on_false = sw.Tensor(sw.Generic([4.0, 5.0, 6.0], dtype=sw.DType.Float32), 0, layout)

    selected = sw.select(condition, on_true, on_false)
    clamped = sw.clamp(on_true, 2.0, 1.0)

    assert [selected[index] for index in range(3)] == [1.0, 5.0, 3.0]
    # Clamp is ordered maximum-then-minimum, with no lower/upper order check.
    assert [clamped[index] for index in range(3)] == [1.0, 1.0, 1.0]
