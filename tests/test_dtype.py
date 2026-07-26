import copy
import inspect
import pickle
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from typing import cast

import pytest

import strideweave as sw
import strideweave.carriers.dtype as dtype_module
from strideweave import (
    CPU,
    BlockScaledDType,
    CompoundDType,
    DType,
    DTypeCategory,
    FileBacked,
    Generic,
    Level,
    SimpleDType,
    SymbolicBits,
    Whole,
    WholeExtent,
)
from strideweave.carriers._dtype import model as dtype_model
from strideweave.carriers.dtype import validate_storage_dtype

BUILT_IN_CATEGORIES = (
    DType.Any,
    DType.Floating,
    DType.Integer,
)
BUILT_IN_SIMPLE_DTYPES = (
    DType.Float32,
    DType.Int32,
    DType.Int8,
    DType.E8M0,
    DType.E5M2,
    DType.E4M3,
    DType.E3M2,
    DType.E2M3,
    DType.E2M1,
)
BUILT_IN_BLOCK_SCALED_DTYPES = (
    DType.MXFP8_E4M3,
    DType.MXFP8_E5M2,
    DType.MXFP6_E3M2,
    DType.MXFP6_E2M3,
    DType.MXFP4,
    DType.MXINT8,
    DType.NVFP4,
)
BUILT_IN_DTYPES = (
    *BUILT_IN_CATEGORIES,
    *BUILT_IN_SIMPLE_DTYPES,
    *BUILT_IN_BLOCK_SCALED_DTYPES,
)


class PlanarDType(CompoundDType, abstract=False):
    """External compound descriptor written with public APIs only.

    This mirrors the extension pattern the README documents: a subclass declares
    ``abstract=False``, hands its planes to ``super().__init__``, and keeps its
    own fields. It never touches the registry, and construction canonicalizes
    and validates the plane mapping before publishing the descriptor.
    """

    __slots__ = ("_label",)

    def __init__(self, name: str, *, planes: object, label: str = "") -> None:
        super().__init__(
            name,
            supertype=DType.Any,
            simple_types=cast(Iterable[SimpleDType], planes),
        )
        self._label = label

    @property
    def label(self) -> str:
        return self._label


class TaggedCompoundDType(CompoundDType, abstract=False):
    """Compound descriptor whose representation carries state of its own.

    ``structure_extension`` is the supported way to describe state beyond the
    common compound contract, so two descriptors that differ only in their tag
    describe different representations.
    """

    __slots__ = ("_tag",)

    def __init__(self, name: str, *, tag: object) -> None:
        super().__init__(name, supertype=DType.Any, simple_types=(DType.Float32,))
        self._tag = tag

    def structure_extension(self) -> tuple[object, ...]:
        return (self._tag,)


def _structure_leaves(structure: object):
    """Yield every non-tuple value reachable in ``structure``."""
    if isinstance(structure, tuple):
        for item in structure:
            yield from _structure_leaves(item)
    else:
        yield structure


def test_dtype_public_api_imports():
    assert sw.DType is DType
    assert sw.DTypeCategory is DTypeCategory
    assert sw.SimpleDType is SimpleDType
    assert issubclass(DTypeCategory, DType)
    assert issubclass(SimpleDType, DType)


def test_dtype_root_is_abstract():
    with pytest.raises(TypeError, match="DType is abstract"):
        DType("Custom")


def test_built_in_categories_are_categories():
    for category in BUILT_IN_CATEGORIES:
        assert isinstance(category, DTypeCategory)
        assert category.is_category()
        assert not category.is_simple()
        assert not category.is_compound()
        assert not hasattr(category, "bits")


def test_built_in_simple_dtypes_declare_exact_bit_widths():
    expected_bits = {
        DType.Float32: 32,
        DType.Int32: 32,
        DType.Int8: 8,
        DType.E8M0: 8,
        DType.E5M2: 8,
        DType.E4M3: 8,
        DType.E3M2: 6,
        DType.E2M3: 6,
        DType.E2M1: 4,
    }
    assert set(expected_bits) == set(BUILT_IN_SIMPLE_DTYPES)
    for dtype, bits in expected_bits.items():
        assert isinstance(dtype, SimpleDType)
        assert dtype.bits == bits
        assert dtype.is_simple()
        assert not dtype.is_category()
        assert not dtype.is_compound()


def test_built_in_supertype_relationships():
    assert DType.Any.supertype is None
    assert DType.Floating.supertype is DType.Any
    assert DType.Integer.supertype is DType.Any
    assert DType.Float32.supertype is DType.Floating
    assert DType.Int32.supertype is DType.Integer

    assert DType.Float32.supertypes() == (DType.Floating, DType.Any)
    assert DType.Int32.supertypes() == (DType.Integer, DType.Any)
    assert DType.Any.supertypes() == ()


@pytest.mark.parametrize(
    ("dtype", "supertype", "expected"),
    [
        (DType.Float32, DType.Float32, True),
        (DType.Float32, DType.Floating, True),
        (DType.Float32, DType.Any, True),
        (DType.Float32, DType.Integer, False),
        (DType.Int32, DType.Integer, True),
        (DType.Int32, DType.Any, True),
        (DType.Int32, DType.Floating, False),
        (DType.Floating, DType.Any, True),
        (DType.Floating, DType.Float32, False),
        (DType.Any, DType.Floating, False),
    ],
)
def test_is_subtype_of_walks_the_category_chain(dtype, supertype, expected):
    assert dtype.is_subtype_of(supertype) is expected


def test_is_subtype_of_rejects_non_dtype_arguments():
    with pytest.raises(TypeError, match="requires a DType"):
        DType.Float32.is_subtype_of("Floating")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "simple", "opaque"),
    [
        (DType.Any, False, True),
        (DType.Floating, False, True),
        (DType.Integer, False, False),
        (DType.Float32, True, False),
        (DType.Int32, True, False),
    ],
)
def test_storage_predicates(dtype, simple, opaque):
    assert dtype.is_simple() is simple
    assert dtype.is_opaque_storage() is opaque


def test_is_simple_classifies_the_representation_not_carrier_support():
    # is_simple() answers whether a dtype is one fixed-width scalar encoding.
    # A registered encoding no carrier accepts is still simple, and so is an
    # extension that no carrier will ever accept.
    unsupported = SimpleDType(
        "TestUnsupportedSimple", bits=24, supertype=DType.Floating
    )

    assert unsupported.is_simple()
    assert not unsupported.is_compound()
    assert not unsupported.is_category()
    assert DType.E4M3.is_simple()
    assert DType.E2M1.is_simple()
    for build in (
        lambda dtype: Generic([1], dtype=dtype),
        lambda dtype: FileBacked(dtype=dtype),
        lambda dtype: CPU(4, dtype=dtype),
    ):
        with pytest.raises(ValueError, match="dtype must be"):
            build(unsupported)


def test_no_descriptor_exposes_a_global_storage_predicate():
    # Storage support is a carrier-class plus dtype decision, never a property
    # of the descriptor: E4M3 is a simple dtype that no carrier accepts today.
    for dtype in BUILT_IN_DTYPES:
        assert not hasattr(dtype, "is_carrier_storable")
    assert DType.E4M3.is_simple()
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=DType.E4M3)


def test_categories_are_never_simple_carrier_storage():
    for category in (DType.Any, DType.Floating, DType.Integer):
        assert not category.is_simple()


def test_registry_returns_registered_singletons_by_class():
    assert set(BUILT_IN_DTYPES) <= set(DType.registered())
    assert set(BUILT_IN_SIMPLE_DTYPES) <= set(SimpleDType.registered())
    assert set(BUILT_IN_CATEGORIES) <= set(DTypeCategory.registered())
    assert set(SimpleDType.registered()).isdisjoint(DTypeCategory.registered())
    assert set(BUILT_IN_BLOCK_SCALED_DTYPES) <= set(BlockScaledDType.registered())
    assert set(BlockScaledDType.registered()).isdisjoint(SimpleDType.registered())


def test_registry_lookup_by_name_is_identity_safe():
    for dtype in BUILT_IN_DTYPES:
        assert DType.from_name(dtype.name) is dtype
    assert SimpleDType.from_name("Float32") is DType.Float32


def test_public_dtype_types_keep_the_facade_module_identity():
    public_types = (
        DType,
        DTypeCategory,
        SimpleDType,
        CompoundDType,
        WholeExtent,
        Level,
        SymbolicBits,
        BlockScaledDType,
    )

    assert all(cls.__module__ == "strideweave.carriers.dtype" for cls in public_types)


def test_registry_lookup_normalizes_names_before_hashing():
    class UnstableName(str):
        def __hash__(self):
            raise RuntimeError("this lookup name must never be hashed")

        def __eq__(self, other):
            raise RuntimeError("this lookup name must never be compared")

    name = UnstableName("Float32")

    assert DType.from_name(name) is DType.Float32
    assert (
        dtype_module._unpickle_dtype(name, DType.Float32.structure()) is DType.Float32
    )


def test_registry_lookup_rejects_non_strings_before_hashing():
    class HostileName:
        def __hash__(self):
            raise RuntimeError("this lookup key must never be hashed")

    name = HostileName()

    with pytest.raises(TypeError, match="lookup name must be a string"):
        DType.from_name(name)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="lookup name must be a string"):
        dtype_module._unpickle_dtype(
            name,  # type: ignore[arg-type]
            DType.Float32.structure(),
        )


def test_value_is_the_read_only_compatibility_alias_for_name():
    # ``DType`` used to be an ``Enum``; ``value`` stays available for callers
    # written against that model and can never diverge from ``name``.
    assert DType.Any.value == "Any"
    assert DType.Floating.value == "Floating"
    assert DType.Float32.value == "Float32"
    assert DType.Int32.value == "Int32"
    for dtype in BUILT_IN_DTYPES:
        assert dtype.value == dtype.name
    with pytest.raises(AttributeError, match="immutable"):
        DType.Float32.value = "Float64"  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del DType.Float32.value  # type: ignore[misc]


def test_carrier_dtype_tags_expose_the_value_alias():
    assert CPU(4).dtype().value == "Float32"
    assert CPU(4, dtype=DType.Int32).dtype().value == "Int32"
    assert Generic([1.0]).dtype().value == "Floating"
    assert Generic([1], dtype=DType.Any).dtype().value == "Any"
    assert FileBacked().dtype().value == "Floating"


def test_registry_lookup_rejects_unknown_and_mistyped_names():
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("Float64")
    with pytest.raises(LookupError, match="not a SimpleDType descriptor"):
        SimpleDType.from_name("Floating")


def test_registering_a_duplicate_name_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        SimpleDType("Float32", bits=32, supertype=DType.Floating)


def test_equality_and_hashing_are_identity_based():
    for dtype in BUILT_IN_DTYPES:
        assert dtype == dtype
        assert hash(dtype) == hash(dtype)
    # These comparisons intentionally exercise the `==`/`!=` semantics that
    # SW002 keeps out of production code: descriptors compare by identity.
    assert DType.Float32 != DType.Int32  # strideweave-lint: ignore=SW002
    assert DType.Float32 != DType.Floating  # strideweave-lint: ignore=SW002
    assert DType.Float32 != "Float32"  # strideweave-lint: ignore=SW002

    tags = {dtype: dtype.name for dtype in BUILT_IN_DTYPES}
    assert len(tags) == len(BUILT_IN_DTYPES)
    assert tags[DType.Float32] == "Float32"


def test_repr_names_the_descriptor_kind_and_name():
    assert repr(DType.Float32) == "SimpleDType('Float32')"
    assert repr(DType.Floating) == "DTypeCategory('Floating')"


def test_descriptors_are_immutable():
    with pytest.raises(AttributeError, match="immutable"):
        DType.Float32.bits = 16  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        DType.Floating.extra = 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="immutable"):
        del DType.Float32.bits  # type: ignore[misc]


def test_copying_and_pickling_preserve_identity():
    for dtype in BUILT_IN_DTYPES:
        assert copy.copy(dtype) is dtype
        assert copy.deepcopy(dtype) is dtype
        assert pickle.loads(pickle.dumps(dtype)) is dtype


def _load_in_fresh_process(
    payload: bytes, *, prelude: str = "", report: str = "repr(loaded)"
) -> str:
    """Unpickle ``payload`` in a new interpreter and report what happened.

    ``prelude`` runs after ``strideweave`` is imported and before the load, which
    is where a receiving process registers the extension dtypes a payload needs.
    ``report`` is an expression over the unpickled ``loaded`` value. Returns that
    expression's value, or ``"<ErrorType>: <message>"`` when the load failed.
    """
    script = (
        "import pickle, sys\n"
        "import strideweave as sw\n"
        f"{prelude}\n"
        "payload = sys.stdin.buffer.read()\n"
        "try:\n"
        "    loaded = pickle.loads(payload)\n"
        f"    print({report})\n"
        "except Exception as error:\n"
        "    print(f'{type(error).__name__}: {error}')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        check=True,
    )
    return completed.stdout.decode().strip()


# The receiving process must define an extension itself, so the child mirrors
# this definition rather than receiving it inside the pickle.
_PICKLED_EXTENSION_DEFINITION = (
    "extension = sw.SimpleDType("
    "'TestPickled16', bits=16, supertype=sw.DType.Floating)\n"
    "block = sw.BlockScaledDType("
    "'TestPickledBlock', element=extension, levels=(sw.Level(sw.DType.E8M0, 32),))"
)


def test_built_in_dtypes_unpickle_in_a_fresh_process():
    # Importing StrideWeave is the only prerequisite for a built-in: its
    # registration happens at import, so the payload resolves to that identity.
    payload = pickle.dumps((DType.Float32, DType.Floating, DType.NVFP4))
    assert (
        _load_in_fresh_process(
            payload,
            report="loaded == (sw.DType.Float32, sw.DType.Floating, sw.DType.NVFP4)",
        )
        == "True"
    )


def test_extension_dtype_unpickles_where_it_is_registered_first():
    extension = SimpleDType("TestPickled16", bits=16, supertype=DType.Floating)
    block = BlockScaledDType(
        "TestPickledBlock", element=extension, levels=(Level(DType.E8M0, 32),)
    )
    payload = pickle.dumps((extension, block))

    # The child resolves both payload entries to the descriptors it registered
    # itself, so identity — not a rebuilt copy — crosses the process boundary.
    assert (
        _load_in_fresh_process(
            payload,
            prelude=_PICKLED_EXTENSION_DEFINITION,
            report="loaded[0] is extension and loaded[1] is block",
        )
        == "True"
    )


def test_extension_dtype_unpickling_requires_a_registration():
    extension = SimpleDType("TestUnregisteredElsewhere", bits=12, supertype=DType.Any)
    payload = pickle.dumps(extension)

    outcome = _load_in_fresh_process(payload)
    assert outcome.startswith("LookupError: ")
    assert "TestUnregisteredElsewhere" in outcome
    assert "must be registered again" in outcome


def test_unpickling_never_substitutes_a_differently_defined_dtype():
    payload = pickle.dumps(
        SimpleDType("TestStructurallyChecked", bits=16, supertype=DType.Floating)
    )

    outcome = _load_in_fresh_process(
        payload,
        prelude=(
            "sw.SimpleDType('TestStructurallyChecked', bits=32, "
            "supertype=sw.DType.Integer)"
        ),
    )
    assert outcome.startswith("ValueError: ")
    assert "TestStructurallyChecked" in outcome
    assert "never substituted" in outcome


# Each case pickles a descriptor in this process and lets a fresh process
# register the same names over a different graph. Only a fingerprint that
# expands every referenced descriptor can tell the two definitions apart.
_COMPOUND_EXTENSION_SOURCE = "\n".join(
    (
        "class Planar(sw.CompoundDType, abstract=False):",
        "    __slots__ = ()",
        "    def __init__(self, name, *, planes):",
        "        super().__init__(",
        "            name, supertype=sw.DType.Any, simple_types=planes",
        "        )",
    )
)
_TAGGED_EXTENSION_SOURCE = "\n".join(
    (
        "class Tagged(sw.CompoundDType, abstract=False):",
        "    __slots__ = ('_tag',)",
        "    def __init__(self, name, *, tag):",
        "        super().__init__(",
        "            name,",
        "            supertype=sw.DType.Any,",
        "            simple_types=(sw.DType.Float32,),",
        "        )",
        "        self._tag = tag",
        "    def structure_extension(self):",
        "        return (self._tag,)",
    )
)
_GRAPH_SUBSTITUTION_CASES = [
    pytest.param(
        lambda: BlockScaledDType(
            "TestGraphElementBlock",
            element=SimpleDType("TestGraphElement", bits=6, supertype=DType.Floating),
            levels=(Level(DType.E8M0, 32),),
        ),
        "\n".join(
            (
                "element = sw.SimpleDType(",
                "    'TestGraphElement', bits=17, supertype=sw.DType.Floating",
                ")",
                "sw.BlockScaledDType(",
                "    'TestGraphElementBlock', element=element,",
                "    levels=(sw.Level(sw.DType.E8M0, 32),),",
                ")",
            )
        ),
        ("17", " 6 "),
        id="element-bits",
    ),
    pytest.param(
        lambda: BlockScaledDType(
            "TestGraphHierarchyBlock",
            element=SimpleDType(
                "TestGraphHierarchyElement", bits=7, supertype=DType.Floating
            ),
            levels=(Level(DType.E8M0, 32),),
        ),
        "\n".join(
            (
                "element = sw.SimpleDType(",
                "    'TestGraphHierarchyElement', bits=7, supertype=sw.DType.Integer",
                ")",
                "sw.BlockScaledDType(",
                "    'TestGraphHierarchyBlock', element=element,",
                "    levels=(sw.Level(sw.DType.E8M0, 32),),",
                ")",
            )
        ),
        ("'Integer'", "'Floating'"),
        id="element-supertype-hierarchy",
    ),
    pytest.param(
        lambda: BlockScaledDType(
            "TestGraphScaleBlock",
            element=DType.E2M1,
            levels=(
                Level(
                    SimpleDType("TestGraphScale", bits=9, supertype=DType.Floating),
                    32,
                ),
            ),
        ),
        "\n".join(
            (
                "scale = sw.SimpleDType(",
                "    'TestGraphScale', bits=9, supertype=sw.DType.Integer",
                ")",
                "sw.BlockScaledDType(",
                "    'TestGraphScaleBlock', element=sw.DType.E2M1,",
                "    levels=(sw.Level(scale, 32),),",
                ")",
            )
        ),
        ("'Integer'", "'Floating'"),
        id="scale-category",
    ),
    pytest.param(
        lambda: SimpleDType(
            "TestGraphTagged",
            bits=11,
            supertype=DTypeCategory(
                "TestGraphCategory", supertype=DType.Any, opaque_storage=True
            ),
        ),
        "\n".join(
            (
                "category = sw.DTypeCategory(",
                "    'TestGraphCategory', supertype=sw.DType.Any, opaque_storage=False",
                ")",
                "sw.SimpleDType('TestGraphTagged', bits=11, supertype=category)",
            )
        ),
        ("False", "True"),
        id="category-opaque-storage",
    ),
    pytest.param(
        lambda: PlanarDType("TestGraphPlanes", planes=(DType.Float32, DType.Int32)),
        "\n".join(
            (
                _COMPOUND_EXTENSION_SOURCE,
                "Planar('TestGraphPlanes', planes=(sw.DType.Float32,"
                " sw.DType.Float32))",
            )
        ),
        ("'Float32'", "'Int32'"),
        id="compound-planes",
    ),
    pytest.param(
        lambda: TaggedCompoundDType("TestGraphTag", tag="row-major"),
        "\n".join(
            (
                _TAGGED_EXTENSION_SOURCE,
                "Tagged('TestGraphTag', tag='column-major')",
            )
        ),
        ("'column-major'", "'row-major'"),
        id="structure-extension",
    ),
]


@pytest.mark.parametrize(("build", "prelude", "fragments"), _GRAPH_SUBSTITUTION_CASES)
def test_unpickling_checks_every_referenced_descriptor(build, prelude, fragments):
    payload = pickle.dumps(build())

    outcome = _load_in_fresh_process(payload, prelude=prelude)

    assert outcome.startswith("ValueError: "), outcome
    assert "never substituted" in outcome, outcome
    for fragment in fragments:
        assert fragment in outcome, outcome


def test_a_matching_referenced_graph_still_resolves_to_the_receiver_identity():
    # The check is structural, not nominal: a receiver that defines the same
    # graph resolves the pickle to its own descriptors.
    element = SimpleDType("TestGraphMatched", bits=5, supertype=DType.Floating)
    block = BlockScaledDType(
        "TestGraphMatchedBlock", element=element, levels=(Level(DType.E8M0, 32),)
    )
    payload = pickle.dumps((element, block))

    prelude = "\n".join(
        (
            "element = sw.SimpleDType(",
            "    'TestGraphMatched', bits=5, supertype=sw.DType.Floating",
            ")",
            "block = sw.BlockScaledDType(",
            "    'TestGraphMatchedBlock', element=element,",
            "    levels=(sw.Level(sw.DType.E8M0, 32),),",
            ")",
        )
    )
    assert (
        _load_in_fresh_process(
            payload,
            prelude=prelude,
            report="loaded[0] is element and loaded[1] is block",
        )
        == "True"
    )


def test_a_pickle_carries_only_a_name_and_a_structure():
    # Nothing in the payload can rebuild a definition: the reducer resolves an
    # identity, and its arguments are the name and the structure to verify.
    reducer, arguments = DType.NVFP4.__reduce__()

    assert reducer is dtype_module._unpickle_dtype
    assert arguments == ("NVFP4", DType.NVFP4.structure())

    leaves = list(_structure_leaves(arguments))
    assert leaves
    assert not any(isinstance(leaf, DType) for leaf in leaves)
    assert all(
        leaf is None or leaf is Whole or type(leaf) in (bool, float, int, str)
        for leaf in leaves
    )


def test_block_scaled_pickles_check_their_element_and_levels():
    payload = pickle.dumps(
        BlockScaledDType(
            "TestCheckedBlock",
            element=DType.E4M3,
            levels=(Level(DType.E8M0, 256),),
        )
    )

    outcome = _load_in_fresh_process(
        payload,
        prelude=(
            "sw.BlockScaledDType('TestCheckedBlock', element=sw.DType.E5M2, "
            "levels=(sw.Level(sw.DType.E8M0, 256),))"
        ),
    )
    assert outcome.startswith("ValueError: ")
    assert "never substituted" in outcome


def test_simple_dtype_construction_validates_width_and_category():
    with pytest.raises(ValueError, match="exact positive bit width"):
        SimpleDType("BadWidth", bits=0, supertype=DType.Floating)
    with pytest.raises(ValueError, match="exact positive bit width"):
        SimpleDType("BadWidthType", bits=True, supertype=DType.Floating)
    with pytest.raises(TypeError, match="must belong to a DTypeCategory"):
        SimpleDType("BadCategory", bits=32, supertype=DType.Float32)  # type: ignore[arg-type]


def test_category_construction_validates_name_and_supertype():
    with pytest.raises(ValueError, match="non-empty string"):
        DTypeCategory("")
    with pytest.raises(TypeError, match="supertype must be a DTypeCategory"):
        DTypeCategory("BadParent", supertype=DType.Float32)  # type: ignore[arg-type]


def test_registered_extension_dtypes_join_the_hierarchy():
    complex_category = DTypeCategory("TestComplex", supertype=DType.Any)
    complex64 = SimpleDType("TestComplex64", bits=64, supertype=complex_category)

    assert complex64.is_simple()
    assert complex64.bits == 64
    assert complex64.is_subtype_of(complex_category)
    assert complex64.is_subtype_of(DType.Any)
    assert not complex64.is_subtype_of(DType.Floating)
    assert SimpleDType.from_name("TestComplex64") is complex64
    assert complex64 in SimpleDType.registered()

    # Every carrier's accepted set is exact, and no extension is in one.
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=complex_category)
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=complex64)


def test_the_opaque_disposition_does_not_extend_any_accepted_set():
    # ``opaque_storage`` records what a legacy category means, not a permission:
    # Generic and FileBacked accept the exact built-in descriptors they
    # document, so an extension declaring the disposition is still rejected.
    opaque = DTypeCategory(
        "TestOpaqueExtension", supertype=DType.Any, opaque_storage=True
    )

    assert opaque.is_opaque_storage()
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=opaque)
    with pytest.raises(ValueError, match="FileBacked dtype must be"):
        FileBacked(dtype=opaque)
    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(4, dtype=opaque)


def test_extensions_are_reached_through_the_registry_not_the_class_namespace():
    # The DType namespace is deliberately the built-in surface only: installing
    # attributes for extensions would let them collide with built-in names and
    # would leave the attribute surface untypable.
    extension = SimpleDType("TestRegistryOnly", bits=24, supertype=DType.Floating)

    assert DType.from_name("TestRegistryOnly") is extension
    assert SimpleDType.from_name("TestRegistryOnly") is extension
    assert extension in SimpleDType.registered()
    assert extension in DType.registered()
    assert not hasattr(DType, "TestRegistryOnly")

    block_scaled = BlockScaledDType(
        "TestRegistryOnlyBlock", element=extension, levels=(Level(DType.E8M0, 8),)
    )
    assert BlockScaledDType.from_name("TestRegistryOnlyBlock") is block_scaled
    assert not hasattr(DType, "TestRegistryOnlyBlock")


def test_dtype_class_namespace_holds_exactly_the_built_in_descriptors():
    exposed = {
        name: value for name, value in vars(DType).items() if isinstance(value, DType)
    }
    assert set(exposed.values()) == set(BUILT_IN_DTYPES)
    for name, dtype in exposed.items():
        assert dtype.name == name
        assert DType.from_name(name) is dtype


def test_carriers_reject_the_integer_category_as_storage():
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=DType.Integer)
    with pytest.raises(ValueError, match="FileBacked dtype must be"):
        FileBacked(dtype=DType.Integer)
    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(4, dtype=DType.Integer)


def test_legacy_opaque_storage_dtypes_remain_supported():
    assert Generic([1.0], dtype=DType.Floating).dtype() is DType.Floating
    assert Generic(["alpha"], dtype=DType.Any).dtype() is DType.Any
    assert FileBacked(dtype=DType.Floating).dtype() is DType.Floating


def test_compound_dtypes_are_not_carrier_storage():
    for dtype in BUILT_IN_BLOCK_SCALED_DTYPES:
        assert isinstance(dtype, CompoundDType)
        assert dtype.is_compound()
        assert not dtype.is_simple()
        assert not dtype.is_category()
        assert not dtype.is_opaque_storage()


def test_abstract_descriptor_classes_cannot_be_constructed():
    # A descriptor that reports dtype semantics without joining the registry is
    # malformed, so abstractness is enforced at the construction boundary,
    # before any argument of a concrete constructor is even considered.
    class BareCompound(CompoundDType):
        __slots__ = ()

    class BareDType(DType):
        __slots__ = ()

    # The metaclass rejects the class itself, so these calls never reach a
    # constructor signature; the checker is told to allow the call anyway.
    abstract_classes = cast(
        tuple[Callable[..., object], ...],
        (DType, CompoundDType, BareCompound, BareDType),
    )
    for abstract_class in abstract_classes:
        with pytest.raises(TypeError, match="is abstract"):
            abstract_class("TestAbstractConstruction", supertype=DType.Any)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestAbstractConstruction")


def test_a_compound_subclass_cannot_redefine_the_owned_plane_accessors():
    # The plane mapping is owned by CompoundDType. An override could serve a
    # live view of mutable state, letting a finalized descriptor's observable
    # representation disagree with the structure recorded for its identity.
    with pytest.raises(TypeError, match="must not redefine simple_types"):

        class OverridingPlanes(CompoundDType, abstract=False):
            __slots__ = ()

            @property
            def simple_types(self) -> tuple[SimpleDType, ...]:
                return (DType.Float32,)

    with pytest.raises(TypeError, match="must not redefine num_carriers"):

        class OverridingCount(CompoundDType, abstract=False):
            __slots__ = ()

            @property
            def num_carriers(self) -> int:
                return 99


def test_a_subclass_cannot_shadow_the_state_the_model_owns():
    # Every hardening above rests on the model owning a descriptor's stored
    # fields and structural accessors. A subclass that redeclared one of them —
    # as a slot, a property, or a plain attribute — could report a
    # representation that disagrees with the recorded structure, so the class
    # is refused when it is created.
    shadowing_definitions = {
        "_name": {"__slots__": ("_name",)},
        "_structure": {"__slots__": ("_structure",)},
        "_finalized": {"_finalized": property(lambda self: False)},
        "_supertype": {"_supertype": property(lambda self: DType.Any)},
        "_bits": {"_bits": 4},
        "name": {"name": property(lambda self: "Float32")},
        "value": {"value": property(lambda self: "Float32")},
        "bits": {"bits": property(lambda self: 4)},
        "structure": {"structure": lambda self: ()},
        "supertypes": {"supertypes": lambda self: ()},
    }

    for member, namespace in shadowing_definitions.items():
        with pytest.raises(TypeError, match=f"must not redefine {member}"):
            type(
                "Shadowing",
                (SimpleDType,),
                {"__slots__": (), **namespace},
                abstract=False,
            )

    for member, namespace in {
        "_element": {"__slots__": ("_element",)},
        "levels": {"levels": property(lambda self: ())},
        "num_axes": {"num_axes": property(lambda self: 0)},
        "bits_per_element": {"bits_per_element": property(lambda self: 0.0)},
    }.items():
        with pytest.raises(TypeError, match=f"must not redefine {member}"):
            type(
                "ShadowingBlock",
                (BlockScaledDType,),
                {"__slots__": (), **namespace},
                abstract=False,
            )

    # The kind predicates are owned by whichever contract class fixes them, so
    # no implementation can claim to be a kind it is not.
    for member in ("is_category", "is_compound", "is_opaque_storage", "is_simple"):
        for base in (DTypeCategory, SimpleDType, BlockScaledDType):
            with pytest.raises(TypeError, match=f"must not redefine {member}"):
                type(
                    f"Shadowing{member}",
                    (base,),
                    {"__slots__": (), member: lambda self: True},
                    abstract=False,
                )


def test_owned_state_cannot_be_patched_onto_a_class_after_it_is_created():
    # The class body is not the only way to shadow owned state: assigning to the
    # class afterwards would do the same, so the rule applies to both.
    class Extension(SimpleDType, abstract=False):
        __slots__ = ()

    for target, member in (
        (Extension, "bits"),
        (Extension, "name"),
        (Extension, "_structure"),
        (SimpleDType, "is_simple"),
        (DType, "structure"),
        (BlockScaledDType, "levels"),
        (CompoundDType, "simple_types"),
    ):
        with pytest.raises(AttributeError, match="owned by the dtype model"):
            setattr(target, member, property(lambda self: None))
        with pytest.raises(AttributeError, match="owned by the dtype model"):
            delattr(target, member)

    assert DType.Float32.bits == 32
    assert DType.MXFP4.levels == (Level(DType.E8M0, 32),)


def test_legacy_ownership_metadata_cannot_disable_the_policy():
    # Ownership was once recorded as class state: replacing
    # CompoundDType._OWNED_MEMBERS with an empty tuple switched the policy off
    # and let simple_types be patched, so DType.MXFP4 reported planes that
    # disagreed with its recorded structure. The policy now lives in module
    # state, so metadata carried on a class is inert however it is written.
    recorded = DType.MXFP4.structure()

    class LegacyMetadata(CompoundDType, abstract=False):
        __slots__ = ()
        _OWNED_MEMBERS = ()

    setattr(CompoundDType, "_OWNED_MEMBERS", ())
    try:
        for target in (CompoundDType, BlockScaledDType, LegacyMetadata):
            for member in ("simple_types", "num_carriers"):
                with pytest.raises(AttributeError, match="owned by the dtype model"):
                    setattr(target, member, property(lambda self: ()))
                with pytest.raises(AttributeError, match="owned by the dtype model"):
                    delattr(target, member)
        with pytest.raises(TypeError, match="must not redefine simple_types"):
            type(
                "LegacyShadowing",
                (LegacyMetadata,),
                {"__slots__": (), "simple_types": property(lambda self: ())},
                abstract=False,
            )
    finally:
        delattr(CompoundDType, "_OWNED_MEMBERS")

    assert DType.MXFP4.simple_types == (DType.E2M1, DType.E8M0)
    assert DType.MXFP4.num_carriers == 2
    assert DType.MXFP4.bits_per_element == 4.25
    assert DType.MXFP4.is_compound() and not DType.MXFP4.is_simple()
    assert DType.MXFP4.structure() == recorded
    assert DType.from_name("MXFP4") is DType.MXFP4
    assert pickle.loads(pickle.dumps(DType.MXFP4)) is DType.MXFP4


def test_ownership_is_layered_over_the_contract_classes():
    # Each contract class owns its own stored fields and structural accessors,
    # and every class below it in the hierarchy inherits that ownership, both in
    # a class body and through a later assignment on the class.
    layers = {
        DType: ("_finalized", "_name", "_structure", "_supertype", "name", "value"),
        DTypeCategory: ("_opaque_storage", "is_category", "is_opaque_storage"),
        SimpleDType: ("_bits", "bits", "is_simple"),
        CompoundDType: ("_simple_types", "is_compound", "num_carriers", "simple_types"),
        BlockScaledDType: ("_element", "bits_per_element", "levels", "num_axes"),
    }
    for contract, members in layers.items():
        for subclass in (other for other in layers if issubclass(other, contract)):
            for member in members:
                with pytest.raises(AttributeError, match="owned by the dtype model"):
                    setattr(subclass, member, property(lambda self: None))
                with pytest.raises(TypeError, match=f"must not redefine {member}"):
                    type(
                        f"Layered{member}",
                        (subclass,),
                        {"__slots__": (), member: property(lambda self: None)},
                        abstract=False,
                    )

    assert DType.Float32.bits == 32
    assert DType.NVFP4.num_axes == 1


def test_an_extension_keeps_owning_the_fields_it_declares():
    # The model's layers are the authority for its own members; an
    # implementation's declared slots stay protected against a further subclass
    # on top of that, so extension storage cannot be shadowed either.
    class Variant(SimpleDType, abstract=False):
        __slots__ = ("_variant",)

    with pytest.raises(TypeError, match="must not redefine _variant"):
        type("DeeperVariant", (Variant,), {"__slots__": ("_variant",)}, abstract=False)


def _dictionary_backed_classes(shadow: Callable[[object], None]):
    """Return one dictionary-backed class per contract, calling ``shadow``.

    None of them declares ``__slots__``, which is supported: their fields live
    in an instance dictionary, and ``shadow`` is what puts an owned name there.
    """

    class ShadowingCategory(DTypeCategory, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any)
            shadow(self)

    class ShadowingSimple(SimpleDType, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, bits=8, supertype=DType.Floating)
            shadow(self)

    class ShadowingCompound(CompoundDType, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any, simple_types=(DType.Float32,))
            shadow(self)

    class ShadowingBlockScaled(BlockScaledDType, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, element=DType.E2M1, levels=(Level(DType.E8M0, 8),))
            shadow(self)

    return (ShadowingCategory, ShadowingSimple, ShadowingCompound, ShadowingBlockScaled)


@pytest.mark.parametrize(
    "member",
    ["structure", "supertypes", "is_subtype_of", "is_simple", "is_compound"],
)
def test_a_descriptor_cannot_assign_an_owned_accessor_on_itself(member):
    # Descriptor state is writable while the constructor runs, and an instance
    # attribute wins over an inherited method, so a descriptor that assigned one
    # would be sealed and published reporting a representation that disagrees
    # with the structure recorded for its identity and carried by its pickle.
    def shadow(descriptor: object) -> None:
        setattr(descriptor, member, lambda *arguments: None)

    for position, shadowing_class in enumerate(_dictionary_backed_classes(shadow)):
        name = f"TestShadowedAccessor{member}{position}"
        with pytest.raises(TypeError, match=f"assigned {member} on itself"):
            shadowing_class(name)
        with pytest.raises(LookupError, match="No StrideWeave dtype named"):
            DType.from_name(name)

    assert DType.Float32.is_simple()
    assert DType.MXFP4.structure() == DType.from_name("MXFP4").structure()


def _reject_shadowed_instance_member(member: str, contract: int, label: str) -> None:
    """Construct a descriptor of one contract whose state shadows ``member``."""

    def shadow(descriptor: object) -> None:
        vars(descriptor)[member] = "spoofed"

    name = f"TestShadowedState{label}{contract}"
    with pytest.raises(TypeError, match=f"assigned {member} on itself"):
        _dictionary_backed_classes(shadow)[contract](name)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name(name)


@pytest.mark.parametrize("member", ["_name", "_structure", "name", "value"])
def test_root_owned_state_reaching_the_instance_dictionary_is_refused(member):
    # Assignment alone cannot shadow a stored field or a property, but an
    # implementation that writes its own dictionary reaches past that. The rule
    # is stated over the descriptor's whole state rather than over one way of
    # populating it, so the root layer holds for every contract.
    for contract in range(4):
        _reject_shadowed_instance_member(member, contract, member)

    assert DType.Float32.name == "Float32"


def test_layer_owned_state_reaching_the_instance_dictionary_is_refused():
    # Below the root, each contract owns what it defines, so the members a
    # descriptor may not shadow are exactly the ones its own layer contributes.
    layers = (
        ("_opaque_storage", "is_opaque_storage"),
        ("_bits", "bits"),
        ("_simple_types", "simple_types", "num_carriers"),
        ("_element", "levels", "num_axes", "bits_per_element"),
    )
    for contract, members in enumerate(layers):
        for member in members:
            _reject_shadowed_instance_member(member, contract, member)

    assert DType.Float32.bits == 32
    assert DType.MXFP4.simple_types == (DType.E2M1, DType.E8M0)
    assert DType.MXFP4.bits_per_element == 4.25


def test_a_rejected_shadowing_descriptor_keeps_no_finalization_state():
    escaped: list[DType] = []

    class Escaping(CompoundDType, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any, simple_types=(DType.Float32,))
            escaped.append(self)
            self.structure = lambda: ()

    with pytest.raises(TypeError, match="assigned structure on itself"):
        Escaping("TestEscapingShadow")

    (rejected,) = escaped
    assert not getattr(rejected, "_finalized", False)
    assert not hasattr(rejected, "_structure")
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestEscapingShadow")


def test_a_shadow_introduced_during_finalization_is_rejected():
    # structure_extension() runs after the constructor and may assign to ``self``
    # as naturally as ``__init__`` does, so a scan that only ran before it would
    # let a descriptor be sealed and registered while its public structure()
    # disagreed with the recorded identity.
    escaped: list[DType] = []

    class LateShadowing(CompoundDType, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any, simple_types=(DType.Float32,))

        def structure_extension(self) -> tuple[object, ...]:
            escaped.append(self)
            setattr(self, "structure", lambda: ("spoofed",))
            return ()

    name = "TestLateShadowExtension"
    with pytest.raises(TypeError, match="assigned structure on itself"):
        LateShadowing(name)

    (rejected,) = escaped
    assert not getattr(rejected, "_finalized", False)
    assert not hasattr(rejected, "_structure")
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name(name)

    # The rejected transaction claimed nothing, so the name is still free.
    replacement = PlanarDType(name, planes=(DType.Float32,))
    assert DType.from_name(name) is replacement


def test_a_late_shadow_leaves_the_structural_key_reusable():
    # A block-scaled descriptor is unique by structure as well as by name, so a
    # rejected one must release both: the representation it described stays
    # available to the descriptor that gets it right.
    class LateShadowingBlock(BlockScaledDType, abstract=False):
        def structure_extension(self) -> tuple[object, ...]:
            vars(self)["num_carriers"] = 99
            return ()

    levels = (Level(DType.E8M0, 64),)
    with pytest.raises(TypeError, match="assigned num_carriers on itself"):
        LateShadowingBlock("TestLateShadowBlock", element=DType.E2M1, levels=levels)

    reused = BlockScaledDType("TestReusedBlock", element=DType.E2M1, levels=levels)
    assert reused.num_carriers == 2
    assert reused.simple_types == (DType.E2M1, DType.E8M0)
    assert pickle.loads(pickle.dumps(reused)) is reused


_IDENTITY_POLICY_HOOKS = (
    "_contract_structure",
    "_structural_key",
    "_structure_conflict",
    "_validate_finalized",
)


@pytest.mark.parametrize("hook", _IDENTITY_POLICY_HOOKS)
def test_an_extension_cannot_own_the_identity_policy_hooks(hook):
    # Validation, the canonical fingerprint layers, structural uniqueness, and
    # the conflict a duplicate reports belong to the contracts a descriptor
    # class inherits, composed by the model over its MRO rather than dispatched
    # through the descriptor. The names the model once dispatched through are
    # therefore reserved: an implementation that defines one is told at class
    # creation that the model does not call it, instead of believing it has
    # changed an identity it has not.
    for base in (DTypeCategory, SimpleDType, CompoundDType, BlockScaledDType):
        with pytest.raises(TypeError, match=f"must not redefine {hook}"):
            type(
                f"Policy{base.__name__}",
                (base,),
                {"__slots__": (), hook: lambda self, *arguments: None},
                abstract=False,
            )
        _reject_mixed_descriptor_class(hook, base)
        with pytest.raises(AttributeError, match="owned by the dtype model"):
            setattr(base, hook, lambda self, *arguments: None)
        with pytest.raises(AttributeError, match="owned by the dtype model"):
            delattr(base, hook)


def test_an_extension_cannot_omit_its_contract_layers_from_its_fingerprint():
    # A compound implementation that described only its own discriminator would
    # drop its planes from the identity two processes compare, so a receiver
    # holding different planes under the same name would be accepted as a match.
    # The layers are the model's, so the attempt is refused and every registered
    # compound descriptor keeps expanding the complete dtype of each plane.
    with pytest.raises(TypeError, match="must not redefine _contract_structure"):

        class Truncating(CompoundDType, abstract=False):
            __slots__ = ()

            def _contract_structure(self) -> tuple[object, ...]:
                return ()

    planar = PlanarDType("TestUnabridgedPlanes", planes=(DType.Float32, DType.Int32))

    assert planar.structure()[2] == "str:CompoundDType"
    assert planar.structure()[3] == (
        ("str:Float32", *DType.Float32.structure()),
        ("str:Int32", *DType.Int32.structure()),
    )


def test_an_extension_cannot_decline_the_uniqueness_its_contract_imposes():
    # Returning no structural key would have made a block-scaled subclass unique
    # by name alone, so two descriptors could claim one representation. The
    # disposition comes from the block-scaled contract, so the subclass is
    # refused and the duplicate representation is still rejected.
    with pytest.raises(TypeError, match="must not redefine _structural_key"):

        class Duplicating(BlockScaledDType, abstract=False):
            __slots__ = ()

            def _structural_key(self) -> object | None:
                return None

    levels = (Level(DType.E8M0, 128),)
    claimed = BlockScaledDType("TestClaimedChain", element=DType.E2M1, levels=levels)

    class Plain(BlockScaledDType, abstract=False):
        __slots__ = ()

    with pytest.raises(ValueError, match="'TestClaimedChain' already describes"):
        Plain("TestDuplicateChain", element=DType.E2M1, levels=levels)

    assert DType.from_name("TestClaimedChain") is claimed
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestDuplicateChain")


def test_an_extension_cannot_weaken_the_validation_it_still_claims():
    # An implementation that overrode finalization validation could pass a check
    # its contract requires without satisfying it — a compound descriptor that
    # never handed its planes to the base, say. The check belongs to the
    # compound contract, so the override is refused and the omission is caught.
    with pytest.raises(TypeError, match="must not redefine _validate_finalized"):

        class Permissive(CompoundDType, abstract=False):
            __slots__ = ()

            def _validate_finalized(self) -> None:
                return None

    class Unplanned(CompoundDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            DType.__init__(self, name, supertype=DType.Any)

    with pytest.raises(TypeError, match="did not initialize its compound base"):
        Unplanned("TestUnvalidatedPlanes")


def test_contract_specifications_cannot_be_weakened_by_class_assignment():
    # The specifications are module state keyed by contract class, not metadata
    # the classes carry, so an implementation cannot present a weaker one: a
    # specification-shaped class attribute is inert whatever it holds, and the
    # composed layers, validation, and uniqueness are unchanged by it.
    recorded = DType.MXFP4.structure()
    inert = dtype_module._ContractSpec(frozenset(), layer=lambda dtype: ())

    class Respecified(BlockScaledDType, abstract=False):
        __slots__ = ()
        _CONTRACT_SPECS = {}
        _contract_spec = inert

    setattr(CompoundDType, "_contract_spec", inert)
    try:
        with pytest.raises(TypeError, match="must not redefine simple_types"):
            type(
                "Respoofing",
                (Respecified,),
                {"__slots__": (), "simple_types": property(lambda self: ())},
                abstract=False,
            )
        with pytest.raises(ValueError, match="'MXFP4' already describes"):
            Respecified(
                "TestRespecifiedMXFP4",
                element=DType.E2M1,
                levels=(Level(DType.E8M0, 32),),
            )
    finally:
        delattr(CompoundDType, "_contract_spec")

    assert DType.MXFP4.structure() == recorded
    assert DType.MXFP4.simple_types == (DType.E2M1, DType.E8M0)


def test_the_canonical_structure_layers_follow_the_contract_mro():
    # The layers a descriptor records are exactly those of the contracts its
    # class inherits, in general-to-specific order, followed by the one
    # contribution the implementation makes. Pinning the composition keeps the
    # format stable for pickles written by other processes.
    assert DType.Floating.structure() == (
        "str:DType",
        ("str:Any", *DType.Any.structure()),
        "str:DTypeCategory",
        "bool:True",
        (),
    )
    assert DType.E2M1.structure() == (
        "str:DType",
        ("str:Floating", *DType.Floating.structure()),
        "str:SimpleDType",
        "int:4",
        (),
    )
    assert DType.MXFP4.structure() == (
        "str:DType",
        ("str:Any", *DType.Any.structure()),
        "str:CompoundDType",
        (
            ("str:E2M1", *DType.E2M1.structure()),
            ("str:E8M0", *DType.E8M0.structure()),
        ),
        "str:BlockScaledDType",
        ("str:E2M1", *DType.E2M1.structure()),
        ((("str:E8M0", *DType.E8M0.structure()), "int:32"),),
        (),
    )

    tagged = TaggedCompoundDType("TestLayeredTag", tag="row-major")
    assert tagged.structure() == (
        "str:DType",
        ("str:Any", *DType.Any.structure()),
        "str:CompoundDType",
        (("str:Float32", *DType.Float32.structure()),),
        ("str:row-major",),
    )


@pytest.mark.parametrize(
    "member", ["__getattribute__", "__getattr__", "__setattr__", "__delattr__"]
)
def test_a_descriptor_class_cannot_intercept_attribute_access(member):
    # Owning the accessors is worth nothing if an implementation can decide what
    # attribute lookup answers: a __getattribute__ reporting 64 for the bits of
    # an 8-bit descriptor would disagree with the int:8 its structure records,
    # and replacing __setattr__ would reopen a sealed descriptor. The attribute
    # machinery is therefore owned exactly like the accessors it serves.
    for base in (DTypeCategory, SimpleDType, CompoundDType, BlockScaledDType):
        with pytest.raises(TypeError, match=f"must not redefine {member}"):
            type(
                f"Intercepting{base.__name__}",
                (base,),
                {"__slots__": (), member: lambda self, *arguments: None},
                abstract=False,
            )
        with pytest.raises(AttributeError, match="owned by the dtype model"):
            setattr(base, member, lambda self, *arguments: None)
        with pytest.raises(AttributeError, match="owned by the dtype model"):
            delattr(base, member)

    assert DType.E4M3.bits == 8
    with pytest.raises(AttributeError, match="descriptors are immutable"):
        setattr(DType.E4M3, "_bits", 64)


def _reject_mixed_descriptor_class(member: str, base: type) -> None:
    """Combine ``base`` with a mixin supplying ``member``, either way round."""
    mixin = type("Mixin", (), {member: lambda self, *arguments: 999})
    for bases in ((mixin, base), (base, mixin)):
        with pytest.raises(TypeError, match=f"inherits {member} from Mixin"):
            type(f"Mixed{base.__name__}", bases, {"__slots__": ()}, abstract=False)


@pytest.mark.parametrize(
    "member", ["__getattribute__", "__getattr__", "structure", "name", "is_subtype_of"]
)
def test_a_descriptor_class_cannot_inherit_owned_members_from_a_mixin(member):
    # What a class body does not define, its bases still supply: a mixin
    # carrying __getattribute__ would report bits == 999 for a descriptor whose
    # structure records int:8. Position in the bases does not make it harmless
    # either, because the model defines no __getattr__ of its own, so the whole
    # initial hierarchy is checked rather than the class body alone.
    for base in (DTypeCategory, SimpleDType, CompoundDType, BlockScaledDType):
        _reject_mixed_descriptor_class(member, base)

    assert DType.E4M3.bits == 8


def test_an_inherited_owned_member_is_refused_at_the_layer_that_owns_it():
    # Below the root, a mixin collides with the contract the class is built on.
    for member, base in (
        ("is_opaque_storage", DTypeCategory),
        ("bits", SimpleDType),
        ("simple_types", CompoundDType),
        ("levels", BlockScaledDType),
    ):
        _reject_mixed_descriptor_class(member, base)

    assert DType.MXFP4.levels == (Level(DType.E8M0, 32),)


def test_a_passive_mixin_and_its_own_api_stay_supported():
    # Only owned members are refused. A mixin that adds behavior of its own is
    # an ordinary base, and the descriptor it helps build registers normally.
    class Describing:
        __slots__ = ()

        def describe(self) -> str:
            descriptor = cast(SimpleDType, self)
            return f"{descriptor.name}/{descriptor.bits}"

    class Described(Describing, SimpleDType, abstract=False):
        __slots__ = ()

    described = Described("TestMixinDescribed", bits=8, supertype=DType.Floating)
    assert described.describe() == "TestMixinDescribed/8"
    assert described.bits == 8
    assert DType.from_name("TestMixinDescribed") is described
    assert pickle.loads(pickle.dumps(described)) is described


def test_an_extension_may_still_define_its_own_properties_and_methods():
    # Only the model's own members are refused; an implementation's API is its
    # own, including names the model never claimed.
    class Described(SimpleDType, abstract=False):
        __slots__ = ("_note",)

        def __init__(self, name: str, *, note: str) -> None:
            super().__init__(name, bits=8, supertype=DType.Floating)
            self._note = note

        @property
        def note(self) -> str:
            return self._note

        def describe(self) -> str:
            return f"{self.name}: {self._note}"

    described = Described("TestDescribedSimple", note="narrow")
    assert described.note == "narrow"
    assert described.describe() == "TestDescribedSimple: narrow"
    assert described.bits == 8
    assert DType.from_name("TestDescribedSimple") is described


def test_a_dictionary_backed_extension_keeps_its_own_fields():
    # Only model-owned names are refused: an implementation that declares no
    # slots still registers, pickles, and reports the canonical accessors.
    class Annotated(SimpleDType, abstract=False):
        def __init__(self, name: str, *, note: str) -> None:
            super().__init__(name, bits=16, supertype=DType.Floating)
            self.note = note

        def structure_extension(self) -> tuple[object, ...]:
            return (self.note,)

    annotated = Annotated("TestAnnotatedSimple", note="documented")
    assert annotated.note == "documented"
    assert annotated.bits == 16
    assert annotated.is_simple()
    assert DType.from_name("TestAnnotatedSimple") is annotated
    assert pickle.loads(pickle.dumps(annotated)) is annotated
    with pytest.raises(AttributeError, match="descriptors are immutable"):
        annotated.note = "changed"


def test_the_supported_extension_immutability_boundary_holds_end_to_end():
    # The supported extension contract in one place: a slotted and a
    # dictionary-backed compound extension both register, keep their own fields,
    # and report accessors that agree with the structure recorded for their
    # identity — through the caller mutating what it passed in, through the
    # implementation's own referenced state changing, and across a pickle. What
    # the model seals is the descriptor, so an object an extension field merely
    # refers to keeps changing, and nothing here promises anything about a class
    # or mixin mutated after the descriptor class is created, which the contract
    # does not support.
    planes = [DType.Float32, DType.Int32]
    revisions: list[str] = ["first"]

    class Slotted(CompoundDType, abstract=False):
        __slots__ = ("_label",)

        def __init__(self, name: str, *, label: str) -> None:
            super().__init__(
                name,
                supertype=DType.Any,
                simple_types=cast(Iterable[SimpleDType], planes),
            )
            self._label = label

        def structure_extension(self) -> tuple[object, ...]:
            return (self._label,)

    class DictionaryBacked(CompoundDType, abstract=False):
        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any, simple_types=(DType.Float32,))
            # An extension field may refer to a mutable object: the model seals
            # the descriptor, not what the descriptor points at, so this stays
            # the implementation's own state and never the recorded identity.
            self.revisions = revisions

    slotted = Slotted("TestBoundarySlotted", label="pair")
    dictionary_backed = DictionaryBacked("TestBoundaryDictionary")
    recorded = {
        descriptor: descriptor.structure()
        for descriptor in (slotted, dictionary_backed)
    }

    planes.append(DType.Int32)
    revisions.append("second")

    for descriptor, structure in recorded.items():
        assert descriptor.structure() == structure
        assert descriptor.is_compound() and not descriptor.is_simple()
        assert descriptor.num_carriers == len(descriptor.simple_types)
        assert DType.from_name(descriptor.name) is descriptor
        assert pickle.loads(pickle.dumps(descriptor)) is descriptor
        assert copy.deepcopy(descriptor) is descriptor
        with pytest.raises(AttributeError, match="descriptors are immutable"):
            setattr(descriptor, "extra", 1)

    assert slotted.simple_types == (DType.Float32, DType.Int32)
    assert dictionary_backed.revisions == ["first", "second"]


@pytest.mark.parametrize("extension", [None, 5, [1], "tag", iter(())])
def test_a_structure_extension_must_return_a_tuple(extension):
    class BadExtensionDType(SimpleDType, abstract=False):
        __slots__ = ()

        def structure_extension(self) -> tuple[object, ...]:
            return cast(tuple[object, ...], extension)

    with pytest.raises(TypeError, match="structure_extension must return a tuple"):
        BadExtensionDType("TestBadExtension", bits=8, supertype=DType.Floating)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestBadExtension")


def test_a_compound_subclass_must_supply_its_planes_to_the_base():
    class UnmappedCompound(CompoundDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            # Omits ``simple_types``, which the compound contract requires.
            DType.__init__(self, name, supertype=DType.Any)

    with pytest.raises(TypeError, match="did not initialize its compound base"):
        UnmappedCompound("TestUnmappedCompound")
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestUnmappedCompound")


def test_a_descriptor_must_be_one_of_the_three_descriptor_kinds():
    # The model is closed: a descriptor reports category, simple, or compound
    # semantics, so a subclass of the root that is none of them is rejected by
    # finalization instead of becoming an unclassifiable registered identity.
    class RootlessDType(DType, abstract=False):
        __slots__ = ()

    with pytest.raises(TypeError, match="not a StrideWeave descriptor kind"):
        RootlessDType("TestUnclassifiable", supertype=DType.Any)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestUnclassifiable")


def test_a_descriptor_implementation_must_initialize_its_base():
    class SkippedBase(SimpleDType, abstract=False):
        __slots__ = ()

        def __init__(self) -> None:
            # Deliberately omits ``super().__init__``, so the descriptor never
            # gets a name and cannot be published under one.
            pass

    with pytest.raises(TypeError, match="did not initialize its DType base"):
        SkippedBase()


def test_every_constructible_compound_descriptor_maps_its_planes():
    for dtype in CompoundDType.registered():
        assert dtype.simple_types
        assert all(plane.is_simple() for plane in dtype.simple_types)
        assert dtype.num_carriers == len(dtype.simple_types)


def test_whole_is_a_terminal_symbolic_extent_singleton():
    assert isinstance(Whole, WholeExtent)
    assert repr(Whole) == "Whole"
    assert sw.Whole is Whole
    assert Level(DType.E8M0, Whole).is_whole()
    assert not Level(DType.E8M0, 32).is_whole()


def test_levels_are_immutable_hashable_values():
    level = Level(DType.E8M0, 32)
    assert level == Level(DType.E8M0, 32)
    assert level != Level(DType.E8M0, 16)
    assert hash(level) == hash(Level(DType.E8M0, 32))
    assert repr(level) == "Level(scale=SimpleDType('E8M0'), block=32)"
    with pytest.raises(AttributeError):
        level.block = 16  # type: ignore[misc]


@pytest.mark.parametrize(
    ("scale", "block", "error", "match"),
    [
        (DType.Floating, 32, TypeError, "scale must be a SimpleDType"),
        (DType.MXFP4, 32, TypeError, "scale must be a SimpleDType"),
        (DType.E8M0, 0, ValueError, "positive integer or Whole"),
        (DType.E8M0, -4, ValueError, "positive integer or Whole"),
        (DType.E8M0, 1.5, TypeError, "integer or Whole"),
        (DType.E8M0, True, TypeError, "integer or Whole"),
    ],
)
def test_level_construction_rejects_invalid_components(scale, block, error, match):
    with pytest.raises(error, match=match):
        Level(scale, block)


@pytest.mark.parametrize(
    ("dtype", "simple_types", "num_axes"),
    [
        (DType.MXFP8_E4M3, (DType.E4M3, DType.E8M0), 1),
        (DType.MXFP8_E5M2, (DType.E5M2, DType.E8M0), 1),
        (DType.MXFP6_E3M2, (DType.E3M2, DType.E8M0), 1),
        (DType.MXFP6_E2M3, (DType.E2M3, DType.E8M0), 1),
        (DType.MXFP4, (DType.E2M1, DType.E8M0), 1),
        (DType.MXINT8, (DType.Int8, DType.E8M0), 1),
        (DType.NVFP4, (DType.E2M1, DType.E4M3, DType.Float32), 1),
    ],
)
def test_block_scaled_carrier_mapping_is_positional_and_total(
    dtype, simple_types, num_axes
):
    assert dtype.simple_types == simple_types
    assert dtype.num_carriers == len(simple_types)
    assert dtype.num_carriers == len(dtype.levels) + 1
    assert dtype.num_axes == num_axes
    assert dtype.element is simple_types[0]
    assert all(plane.is_simple() for plane in dtype.simple_types)


def test_mx_formats_use_the_spec_mandated_block_extent():
    for dtype in (
        DType.MXFP8_E4M3,
        DType.MXFP8_E5M2,
        DType.MXFP6_E3M2,
        DType.MXFP6_E2M3,
        DType.MXFP4,
        DType.MXINT8,
    ):
        assert dtype.levels == (Level(DType.E8M0, 32),)

    assert DType.NVFP4.levels == (
        Level(DType.E4M3, 16),
        Level(DType.Float32, Whole),
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (DType.MXFP8_E4M3, 8.25),
        (DType.MXFP8_E5M2, 8.25),
        (DType.MXFP6_E3M2, 6.25),
        (DType.MXFP6_E2M3, 6.25),
        (DType.MXFP4, 4.25),
        (DType.MXINT8, 8.25),
    ],
)
def test_concrete_bits_per_element_matches_the_specification(dtype, expected):
    assert dtype.bits_per_element == expected


def test_whole_level_yields_symbolic_bits_per_element():
    symbolic = DType.NVFP4.bits_per_element
    assert isinstance(symbolic, SymbolicBits)
    assert symbolic.constant == 4.5
    assert symbolic.whole_scale_bits == 32
    assert symbolic.evaluate(32) == 4.5 + 32 / 32
    assert symbolic.evaluate(1024) == 4.5 + 32 / 1024

    with pytest.raises(ValueError, match="must be positive"):
        symbolic.evaluate(0)
    with pytest.raises(TypeError, match="must be an integer"):
        symbolic.evaluate(1024.0)  # type: ignore[arg-type]


def test_block_size_accounting_matches_the_specification_cross_check():
    # A 32-element MXFP8 block occupies 264 bits; an MXFP4 block occupies 136.
    mxfp8_bits = DType.MXFP8_E4M3.bits_per_element
    mxfp4_bits = DType.MXFP4.bits_per_element
    assert isinstance(mxfp8_bits, float)
    assert isinstance(mxfp4_bits, float)
    assert mxfp8_bits * 32 == 264
    assert mxfp4_bits * 32 == 136


def test_block_scaled_construction_rejects_invalid_structures():
    with pytest.raises(TypeError, match="element dtype must be a SimpleDType"):
        BlockScaledDType(
            "TestBadElement",
            element=DType.Floating,  # type: ignore[arg-type]
            levels=(Level(DType.E8M0, 32),),
        )
    with pytest.raises(ValueError, match="at least one scale level"):
        BlockScaledDType("TestNoLevels", element=DType.E2M1, levels=())
    with pytest.raises(TypeError, match="must be Level descriptors"):
        BlockScaledDType(
            "TestBadLevel",
            element=DType.E2M1,
            levels=(DType.E8M0,),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Only the final block-scaled level"):
        BlockScaledDType(
            "TestNonTerminalWhole",
            element=DType.E2M1,
            levels=(Level(DType.Float32, Whole), Level(DType.E8M0, 32)),
        )


def test_block_scaled_descriptors_are_unique_by_structure():
    with pytest.raises(ValueError, match="'MXFP4' already describes"):
        BlockScaledDType(
            "TestDuplicateOfMXFP4",
            element=DType.E2M1,
            levels=(Level(DType.E8M0, 32),),
        )


def test_block_scaled_descriptors_are_immutable_and_well_represented():
    assert repr(DType.MXFP4) == "BlockScaledDType('MXFP4')"
    assert DType.MXFP4.name == "MXFP4"
    assert DType.from_name("MXFP4") is DType.MXFP4
    assert BlockScaledDType.from_name("NVFP4") is DType.NVFP4
    assert hash(DType.MXFP4) == hash(DType.MXFP4)
    assert copy.deepcopy(DType.NVFP4) is DType.NVFP4
    assert pickle.loads(pickle.dumps(DType.NVFP4)) is DType.NVFP4
    with pytest.raises(AttributeError, match="immutable"):
        DType.MXFP4.element = DType.E4M3  # type: ignore[misc]


def test_block_scaled_descriptors_sit_under_the_root_category():
    for dtype in BUILT_IN_BLOCK_SCALED_DTYPES:
        assert dtype.supertype is DType.Any
        assert dtype.is_subtype_of(DType.Any)
        assert not dtype.is_subtype_of(DType.Floating)


def test_no_dtype_constructor_accepts_a_format_mandated_block_size():
    # A block extent belongs to a format's definition, not to its caller, so of
    # the dtype constructors and factories only the structural Level definition
    # used to build the registry names one. The extents the built-in formats
    # mandate are checked in test_mx_formats_use_the_spec_mandated_block_extent.
    block_parameters = {"block", "block_size", "blocks", "block_extent"}
    callables = {
        name for name in dtype_module.__all__ if callable(getattr(dtype_module, name))
    }
    assert callables == {
        "BlockScaledDType",
        "CompoundDType",
        "DType",
        "DTypeCategory",
        "Level",
        "SimpleDType",
        "SymbolicBits",
        "WholeExtent",
        "validate_storage_dtype",
    }

    for name in callables - {"Level"}:
        parameters = set(inspect.signature(getattr(dtype_module, name)).parameters)
        assert not parameters & block_parameters, (
            f"strideweave.carriers.dtype.{name} must not accept a block size"
        )
    assert "block" in inspect.signature(Level).parameters


def test_registered_block_scaled_extension_joins_the_hierarchy():
    fp8_block_16 = BlockScaledDType(
        "TestFP8Block16",
        element=DType.E4M3,
        levels=(Level(DType.E8M0, 16),),
    )

    assert fp8_block_16.simple_types == (DType.E4M3, DType.E8M0)
    assert fp8_block_16.num_carriers == 2
    assert fp8_block_16.num_axes == 1
    assert fp8_block_16.bits_per_element == 8.5
    assert BlockScaledDType.from_name("TestFP8Block16") is fp8_block_16
    assert fp8_block_16 is not DType.MXFP8_E4M3


def test_supported_carrier_construction_remains_compatible():
    assert Generic([1.0]).dtype() is DType.Floating
    assert Generic([1.0], dtype=DType.Any).dtype() is DType.Any
    assert CPU(4).dtype() is DType.Float32
    assert CPU(4, dtype=DType.Int32).dtype() is DType.Int32
    assert FileBacked().dtype() is DType.Floating
    assert FileBacked(dtype=DType.Float32).dtype() is DType.Float32
    assert FileBacked(dtype=DType.Int32).dtype() is DType.Int32


@pytest.mark.parametrize(
    ("carrier", "expected"),
    [
        (lambda dtype: Generic([1], dtype=dtype), "Generic cannot store compound"),
        (lambda dtype: FileBacked(dtype=dtype), "FileBacked cannot store compound"),
    ],
)
@pytest.mark.parametrize("dtype", BUILT_IN_BLOCK_SCALED_DTYPES)
def test_compound_dtypes_are_rejected_before_any_storage_is_built(
    carrier, expected, dtype
):
    with pytest.raises(ValueError, match=expected) as raised:
        carrier(dtype)
    assert "one carrier per simple_types plane" in str(raised.value)
    assert dtype.name in str(raised.value)


@pytest.mark.parametrize("dtype", BUILT_IN_BLOCK_SCALED_DTYPES)
def test_cpu_rejects_compound_dtypes_before_allocation(dtype):
    with pytest.raises(ValueError, match="CPU cannot store compound") as raised:
        CPU(4, dtype=dtype)
    message = str(raised.value)
    assert dtype.name in message
    assert "one carrier per simple_types plane" in message


def test_every_carrier_explains_deferred_compound_storage_the_same_way():
    # RT012: the native CPU parser and validate_storage_dtype agree in meaning.
    messages = []
    for build in (
        lambda: Generic([1], dtype=DType.MXFP4),
        lambda: FileBacked(dtype=DType.MXFP4),
        lambda: CPU(4, dtype=DType.MXFP4),
    ):
        with pytest.raises(ValueError, match="cannot store compound dtype") as raised:
            build()
        messages.append(str(raised.value))

    for carrier, message in zip(
        ("Generic", "FileBacked", "CPU"), messages, strict=True
    ):
        assert message.startswith(f"{carrier} cannot store compound dtype 'MXFP4': ")
        assert message.endswith(
            "a carrier holds one simple dtype, and a compound representation "
            "needs one carrier per simple_types plane, which is not implemented"
        )


def test_each_carrier_accepts_exactly_its_documented_dtype_set():
    # RT012 states each accepted set exactly, so it is checked exhaustively
    # against every built-in descriptor rather than by sampling. Note that the
    # sets are not "the simple dtypes": Generic accepts only the two legacy
    # opaque categories, and the narrow simple encodings are accepted nowhere.
    accepted_sets = {
        "Generic": (DType.Any, DType.Floating),
        "CPU": (DType.Float32, DType.Int32),
        "FileBacked": (DType.Floating, DType.Float32, DType.Int32),
    }
    builders = {
        "Generic": lambda dtype: Generic([1], dtype=dtype),
        "CPU": lambda dtype: CPU(4, dtype=dtype),
        "FileBacked": lambda dtype: FileBacked(dtype=dtype),
    }

    for carrier, build in builders.items():
        accepted = accepted_sets[carrier]
        for dtype in BUILT_IN_DTYPES:
            if any(dtype is candidate for candidate in accepted):
                assert build(dtype).dtype() is dtype
                continue
            with pytest.raises(ValueError, match=f"^{carrier} "):
                build(dtype)


class _SpoofedDType:
    """Object that claims equality with anything and records every comparison."""

    def __init__(self) -> None:
        self.compared: list[object] = []

    def __eq__(self, other: object) -> bool:
        self.compared.append(other)
        return True

    def __hash__(self) -> int:
        return 0


class _ExplodingDType:
    """Object whose equality must never be reached."""

    def __eq__(self, other: object) -> bool:
        raise AssertionError("a carrier must not consult dtype equality")

    def __hash__(self) -> int:
        return 0


def test_carriers_recognize_storage_dtypes_by_identity_not_equality():
    # RT012 matches each accepted set by identity, and the native CPU parser
    # mirrors it. An object that merely compares equal to DType.Float32 is not
    # that dtype, and equality is never consulted to find out.
    builders = (
        lambda dtype: Generic([1], dtype=dtype),
        lambda dtype: FileBacked(dtype=dtype),
        lambda dtype: CPU(4, dtype=dtype),
    )

    for build in builders:
        spoofed = _SpoofedDType()
        with pytest.raises((TypeError, ValueError)) as raised:
            build(spoofed)  # type: ignore[arg-type]
        assert "dtype must be" in str(raised.value)
        assert spoofed.compared == []

        with pytest.raises((TypeError, ValueError)) as raised:
            build(_ExplodingDType())  # type: ignore[arg-type]
        assert "dtype must be" in str(raised.value)


def test_a_dtype_lookalike_cannot_impersonate_a_cpu_storage_dtype():
    # A distinct SimpleDType named and shaped exactly like Float32 cannot be
    # registered, so the closest an impostor gets is an unregistered object that
    # mimics the descriptor surface. The native parser still refuses it.
    class Float32Lookalike:
        name = "Float32"
        value = "Float32"
        bits = 32

        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return 0

        def is_compound(self) -> bool:
            return False

    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(4, dtype=Float32Lookalike())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(4).allocate_like(4, dtype=Float32Lookalike())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(4).new_like([1.0], dtype=Float32Lookalike())  # type: ignore[arg-type]


def test_cpu_allocation_round_trips_the_canonical_dtype_singletons():
    for dtype in (DType.Float32, DType.Int32):
        carrier = CPU(4, dtype=dtype)
        assert carrier.dtype() is dtype
        assert carrier.allocate_like(2).dtype() is dtype
        assert carrier.allocate_like(2, dtype=dtype).dtype() is dtype
        assert carrier.new_like([1]).dtype() is dtype
    assert CPU(4).dtype() is DType.Float32
    assert CPU(4, dtype=None).dtype() is DType.Float32  # type: ignore[arg-type]
    assert CPU(4, dtype=DType.Float32).allocate_like(2, dtype=DType.Int32).dtype() is (
        DType.Int32
    )


@pytest.mark.parametrize(
    "dtype",
    [DType.Integer, DType.E4M3, DType.E8M0, DType.Int8, DType.E2M1],
)
def test_carriers_reject_dtypes_without_storage_support(dtype):
    with pytest.raises(ValueError, match="Generic dtype must be"):
        Generic([1], dtype=dtype)
    with pytest.raises(ValueError, match="FileBacked dtype must be"):
        FileBacked(dtype=dtype)
    with pytest.raises(ValueError, match="CPU dtype must be"):
        CPU(4, dtype=dtype)


def test_validate_storage_dtype_reports_accepted_alternatives():
    assert (
        validate_storage_dtype(
            DType.Float32, carrier="CPU", accepted=(DType.Float32, DType.Int32)
        )
        is DType.Float32
    )
    with pytest.raises(TypeError, match="Demo dtype must be a DType"):
        validate_storage_dtype("Float32", carrier="Demo", accepted=(DType.Float32,))
    with pytest.raises(ValueError, match=r"Demo dtype must be DType\.Float32$"):
        validate_storage_dtype(DType.Int32, carrier="Demo", accepted=(DType.Float32,))
    with pytest.raises(
        ValueError, match=r"Demo dtype must be DType\.Float32 or DType\.Int32$"
    ):
        validate_storage_dtype(
            DType.Int8, carrier="Demo", accepted=(DType.Float32, DType.Int32)
        )
    with pytest.raises(
        ValueError,
        match=r"Demo dtype must be DType\.Floating, DType\.Float32, or DType\.Int32$",
    ):
        validate_storage_dtype(
            DType.Int8,
            carrier="Demo",
            accepted=(DType.Floating, DType.Float32, DType.Int32),
        )


def test_evictable_composition_preserves_dtype_matching():
    hierarchy = sw.Evictable(CPU(2, dtype=DType.Int32), FileBacked(dtype=DType.Int32))
    assert hierarchy.dtype() is DType.Int32

    with pytest.raises(TypeError, match="primary and secondary dtypes must match"):
        sw.Evictable(CPU(2, dtype=DType.Int32), FileBacked(dtype=DType.Float32))


def test_autograd_participation_is_the_documented_floating_pair():
    layout = sw.Layout(sw.Shape(1), sw.Stride(1))
    for carrier in (Generic([1.0]), CPU(1)):
        tensor = sw.Tensor(carrier, 0, layout)
        assert tensor.dtype() in (DType.Floating, DType.Float32)
        assert tensor.is_differentiable() is True

    for carrier in (Generic([1], dtype=DType.Any), CPU(1, dtype=DType.Int32)):
        assert sw.Tensor(carrier, 0, layout).is_differentiable() is False


def test_failed_registration_reserves_neither_name_nor_structure():
    novel_levels = (Level(DType.E8M0, 64),)

    with pytest.raises(ValueError, match="already registered"):
        BlockScaledDType("MXFP4", element=DType.E4M3, levels=novel_levels)

    # The rejected structure was never reserved, so it still registers cleanly.
    recovered = BlockScaledDType(
        "TestRecoveredStructure", element=DType.E4M3, levels=novel_levels
    )
    assert recovered.levels == novel_levels
    assert BlockScaledDType.from_name("TestRecoveredStructure") is recovered
    assert DType.MXFP4.element is DType.E2M1


def test_a_failed_commit_releases_the_structure_it_had_already_claimed(monkeypatch):
    # The structure and the name are two insertions, so the second one failing
    # must not leave the representation reserved by a descriptor that never
    # became reachable. The failure is injected rather than raced, so the
    # window is exercised deterministically.
    novel_levels = (Level(DType.E8M0, 96),)

    class FailingRegistry(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("registry commit failed")

    monkeypatch.setattr(
        dtype_model, "_REGISTRY", FailingRegistry(dtype_model._REGISTRY)
    )
    with pytest.raises(RuntimeError, match="registry commit failed"):
        BlockScaledDType(
            "TestInterruptedCommit", element=DType.E4M3, levels=novel_levels
        )
    monkeypatch.undo()

    # Neither the name nor the structure was left claimed, so a later valid
    # descriptor can take both.
    recovered = BlockScaledDType(
        "TestInterruptedCommit", element=DType.E4M3, levels=novel_levels
    )
    assert BlockScaledDType.from_name("TestInterruptedCommit") is recovered
    assert recovered.levels == novel_levels


def test_an_external_compound_extension_registers_through_public_apis_only():
    # The extension imports nothing private: there is no module-level registry
    # helper for it to reach for, and construction publishes the descriptor.
    assert not hasattr(dtype_module, "_register")
    assert not hasattr(dtype_module, "_register_block_scaled")

    planar = PlanarDType("TestPlanarPair", planes=(DType.Float32, DType.Int32))

    assert planar.is_compound()
    assert not planar.is_simple()
    assert planar.simple_types == (DType.Float32, DType.Int32)
    assert planar.num_carriers == 2
    assert planar.supertype is DType.Any
    assert DType.from_name("TestPlanarPair") is planar
    assert CompoundDType.from_name("TestPlanarPair") is planar
    assert planar in CompoundDType.registered()
    assert copy.copy(planar) is planar
    assert copy.deepcopy(planar) is planar
    assert pickle.loads(pickle.dumps(planar)) is planar
    with pytest.raises(AttributeError, match="immutable"):
        planar.planes = ()  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="Generic cannot store compound"):
        Generic([1], dtype=planar)


# The receiving process defines the extension itself, exactly as it must for a
# simple or block-scaled extension: a pickle identifies a descriptor, never a
# definition to rebuild.
_PICKLED_COMPOUND_DEFINITION = "\n".join(
    (
        _COMPOUND_EXTENSION_SOURCE,
        "planar = Planar(",
        "    'TestPlanarPickled', planes=(sw.DType.Float32, sw.DType.Int32)",
        ")",
    )
)


def test_an_external_compound_extension_unpickles_where_it_is_registered_first():
    planar = PlanarDType("TestPlanarPickled", planes=(DType.Float32, DType.Int32))
    payload = pickle.dumps(planar)

    assert (
        _load_in_fresh_process(
            payload,
            prelude=_PICKLED_COMPOUND_DEFINITION,
            report="loaded is planar",
        )
        == "True"
    )


def test_a_constructor_failing_after_its_base_registers_nothing():
    # A most-derived constructor that raises must not leave a ghost behind: the
    # name it initialized under stays free for a later, valid descriptor, and so
    # does any structure it would have claimed.
    class FailingCategory(DTypeCategory, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any)
            raise RuntimeError("constructor failed after its base initialized")

    class FailingSimple(SimpleDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            super().__init__(name, bits=8, supertype=DType.Floating)
            raise RuntimeError("constructor failed after its base initialized")

    class FailingCompound(PlanarDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            super().__init__(name, planes=(DType.Float32,))
            raise RuntimeError("constructor failed after its base initialized")

    class FailingBlockScaled(BlockScaledDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str, levels: tuple[Level, ...]) -> None:
            super().__init__(name, element=DType.E4M3, levels=levels)
            raise RuntimeError("constructor failed after its base initialized")

    ghost_levels = (Level(DType.E8M0, 48),)
    cases = (
        (
            "TestGhostCategory",
            FailingCategory,
            lambda name: DTypeCategory(name, supertype=DType.Any),
        ),
        (
            "TestGhostSimple",
            FailingSimple,
            lambda name: SimpleDType(name, bits=8, supertype=DType.Floating),
        ),
        (
            "TestGhostCompound",
            FailingCompound,
            lambda name: PlanarDType(name, planes=(DType.Float32,)),
        ),
        (
            "TestGhostBlockScaled",
            lambda name: FailingBlockScaled(name, ghost_levels),
            lambda name: BlockScaledDType(
                name, element=DType.E4M3, levels=ghost_levels
            ),
        ),
    )

    for name, failing, recover in cases:
        with pytest.raises(RuntimeError, match="failed after its base initialized"):
            failing(name)
        with pytest.raises(LookupError, match="No StrideWeave dtype named"):
            DType.from_name(name)

        # The retry proves neither the name nor the structure stayed reserved.
        recovered = recover(name)
        assert DType.from_name(name) is recovered


@pytest.mark.parametrize(
    ("planes", "error", "match"),
    [
        ((), ValueError, "at least one representation plane"),
        ([], ValueError, "at least one representation plane"),
        (iter(()), ValueError, "at least one representation plane"),
        (DType.Float32, TypeError, "must be an iterable of SimpleDType"),
        (None, TypeError, "must be an iterable of SimpleDType"),
        ((DType.Floating,), TypeError, r"simple_types\[0\] must be a SimpleDType"),
        (
            (DType.Float32, DType.MXFP4),
            TypeError,
            r"simple_types\[1\] must be a SimpleDType",
        ),
        ("Float32", TypeError, r"simple_types\[0\] must be a SimpleDType"),
    ],
)
def test_compound_planes_are_validated_before_any_registry_mutation(
    planes, error, match
):
    with pytest.raises(error, match=match):
        PlanarDType("TestInvalidPlanes", planes=planes)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestInvalidPlanes")


@pytest.mark.parametrize(
    "supplied",
    [
        lambda planes: planes,
        list,
        iter,
        lambda planes: (plane for plane in planes),
    ],
)
def test_compound_planes_are_copied_out_of_the_callers_collection(supplied):
    # Whatever iterable the caller passes, the descriptor keeps its own tuple.
    planes = [DType.Float32, DType.Int32]
    descriptor = PlanarDType(
        f"TestCopiedPlanes{id(supplied):x}", planes=supplied(planes)
    )

    assert type(descriptor.simple_types) is tuple
    assert descriptor.simple_types == (DType.Float32, DType.Int32)
    assert descriptor.num_carriers == 2


def test_mutating_the_supplied_collection_cannot_change_a_registered_descriptor():
    # The adversarial case: the caller keeps the list it passed in and mutates
    # it afterwards. A descriptor that served a live view of that list would
    # report planes its recorded structure, registry key, and pickle disagree
    # with, so the model copies the mapping instead.
    planes = [DType.Float32, DType.Int32]
    descriptor = PlanarDType("TestAliasedPlanes", planes=planes)
    structure = descriptor.structure()
    payload = pickle.dumps(descriptor)

    planes.append(DType.Int8)
    planes[0] = DType.Int8
    planes.clear()

    assert descriptor.simple_types == (DType.Float32, DType.Int32)
    assert descriptor.num_carriers == 2
    assert descriptor.structure() == structure
    assert hash(descriptor) == hash(DType.from_name("TestAliasedPlanes"))
    assert DType.from_name("TestAliasedPlanes") is descriptor
    assert pickle.loads(payload) is descriptor
    assert pickle.loads(pickle.dumps(descriptor)) is descriptor
    with pytest.raises(AttributeError, match="immutable"):
        descriptor.simple_types = ()  # type: ignore[misc]


def test_a_block_scaled_subclass_is_not_a_second_identity_for_a_representation():
    # Structure uniqueness is anchored at the block-scaled contract, so a
    # subclass that adds no structure describes an already registered
    # representation rather than a new one.
    class TaggedBlockScaled(BlockScaledDType, abstract=False):
        __slots__ = ()

    with pytest.raises(ValueError, match="'MXFP4' already describes"):
        TaggedBlockScaled(
            "TestSubclassedMXFP4",
            element=DType.E2M1,
            levels=(Level(DType.E8M0, 32),),
        )
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestSubclassedMXFP4")


def test_a_partly_constructed_descriptor_is_never_reachable():
    # Publication happens after the most-derived constructor returns, so a
    # concurrent lookup cannot observe a descriptor mid-initialization.
    started = threading.Event()
    release = threading.Event()

    class SlowSimpleDType(SimpleDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            super().__init__(name, bits=8, supertype=DType.Floating)
            started.set()
            release.wait(timeout=10)

    constructed: list[object] = []
    thread = threading.Thread(
        target=lambda: constructed.append(SlowSimpleDType("TestSlowlyBuilt"))
    )
    thread.start()
    try:
        assert started.wait(timeout=10)
        with pytest.raises(LookupError, match="No StrideWeave dtype named"):
            DType.from_name("TestSlowlyBuilt")
    finally:
        release.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert DType.from_name("TestSlowlyBuilt") is constructed[0]


def test_a_structure_extension_distinguishes_otherwise_equal_representations():
    # Two tagged descriptors share every field of the compound contract, so only
    # the extension keeps their representations — and their pickles — apart.
    first = TaggedCompoundDType("TestTaggedFirst", tag="row-major")
    second = TaggedCompoundDType("TestTaggedSecond", tag="column-major")

    assert first.simple_types == second.simple_types == (DType.Float32,)
    assert first.structure() != second.structure()
    assert first.structure_extension() == ("row-major",)
    assert pickle.loads(pickle.dumps(first)) is first


def test_a_structure_extension_can_claim_a_distinct_block_scaled_representation():
    # A block-scaled subclass that genuinely adds structure describes its own
    # representation, so it may reuse an element and level chain already
    # claimed by the built-in format.
    class VariantBlockScaled(BlockScaledDType, abstract=False):
        __slots__ = ("_variant",)

        def __init__(
            self,
            name: str,
            *,
            variant: str,
            element: SimpleDType,
            levels: tuple[Level, ...],
        ) -> None:
            super().__init__(name, element=element, levels=levels)
            self._variant = variant

        def structure_extension(self) -> tuple[object, ...]:
            return (self._variant,)

    variant = VariantBlockScaled(
        "TestVariantOfMXFP4",
        variant="saturating",
        element=DType.E2M1,
        levels=(Level(DType.E8M0, 32),),
    )

    assert BlockScaledDType.from_name("TestVariantOfMXFP4") is variant
    assert variant is not DType.MXFP4
    assert variant.simple_types == DType.MXFP4.simple_types
    assert variant.structure() != DType.MXFP4.structure()

    # The variant's own representation is claimed exactly once.
    with pytest.raises(ValueError, match="'TestVariantOfMXFP4' already describes"):
        VariantBlockScaled(
            "TestSecondVariantOfMXFP4",
            variant="saturating",
            element=DType.E2M1,
            levels=(Level(DType.E8M0, 32),),
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (True, 1),
        (False, 0),
        (1, 1.0),
        (True, 1.0),
        ("1", 1),
        (0.0, -0.0),
        (None, "none"),
        (Whole, "whole"),
    ],
)
def test_numerically_equal_tags_of_different_types_stay_distinct(first, second):
    # Structures decide identity, so they cannot compare through ordinary Python
    # equality: True == 1 == 1.0 would collapse three different representations
    # into one, and a sender's tag could then reconstruct against a receiver's.
    left = TaggedCompoundDType(f"TestTagLeft{first!r}{second!r}", tag=first)
    right = TaggedCompoundDType(f"TestTagRight{first!r}{second!r}", tag=second)

    assert left.structure() != right.structure()
    assert pickle.loads(pickle.dumps(left)) is left
    assert pickle.loads(pickle.dumps(right)) is right


def test_a_sender_tag_cannot_reconstruct_against_a_differently_typed_receiver_tag():
    payload = pickle.dumps(TaggedCompoundDType("TestTypedTag", tag=True))

    outcome = _load_in_fresh_process(
        payload,
        prelude="\n".join((_TAGGED_EXTENSION_SOURCE, "Tagged('TestTypedTag', tag=1)")),
    )

    assert outcome.startswith("ValueError: "), outcome
    assert "never substituted" in outcome, outcome
    assert " 1 here and True in the pickle" in outcome, outcome


@pytest.mark.parametrize(
    ("tag", "described"),
    [
        (0.1, "0.1"),
        (-0.0, "-0.0"),
        (2.0**-1074, "5e-324"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
    ],
)
def test_finite_and_infinite_float_tags_reconstruct_exactly(tag, described):
    # Floats are recorded in exact hexadecimal form, so a tag round-trips
    # without precision loss and reconstructs in another process.
    descriptor = TaggedCompoundDType(f"TestFloatTag{described}", tag=tag)
    payload = pickle.dumps(descriptor)

    assert pickle.loads(payload) is descriptor
    assert (
        _load_in_fresh_process(
            payload,
            prelude="\n".join(
                (
                    _TAGGED_EXTENSION_SOURCE,
                    f"tagged = Tagged('TestFloatTag{described}', tag=float.fromhex("
                    f"{tag.hex()!r}))",
                )
            ),
            report="loaded is tagged",
        )
        == "True"
    )


def test_a_nan_tag_is_canonical_rather_than_non_reflexive():
    # NaN is not equal to itself, so a structure holding it would fail even
    # same-process reconstruction if it were compared numerically. The encoding
    # gives it one deterministic spelling instead.
    nan = float("nan")
    descriptor = TaggedCompoundDType("TestNanTag", tag=nan)

    assert not nan == nan  # the non-reflexivity the encoding has to absorb
    assert descriptor.structure() == descriptor.structure()
    assert pickle.loads(pickle.dumps(descriptor)) is descriptor
    assert (
        _load_in_fresh_process(
            pickle.dumps(descriptor),
            prelude="\n".join(
                (
                    _TAGGED_EXTENSION_SOURCE,
                    "tagged = Tagged('TestNanTag', tag=float('nan'))",
                )
            ),
            report="loaded is tagged",
        )
        == "True"
    )

    # A NaN tag is still a different representation from every number.
    assert (
        TaggedCompoundDType("TestNanTagPeer", tag=1.0).structure()
        != descriptor.structure()
    )


def test_every_structure_leaf_is_an_encoded_string():
    # The encoding is what makes comparison type-exact, so no raw value — not
    # even a descriptor name or a block extent — is stored unencoded.
    for dtype in (*BUILT_IN_DTYPES, TaggedCompoundDType("TestEncodedLeaves", tag=7)):
        leaves = list(_structure_leaves(dtype.structure()))
        assert leaves
        assert all(type(leaf) is str for leaf in leaves), dtype.name


@pytest.mark.parametrize(
    "extension",
    [
        (DType.Float32,),
        ([1, 2],),
        ({"tag": "row-major"},),
        (Level(DType.E8M0, 32),),
        (b"bytes",),
        (complex(1, 2),),
        ((1, [2]),),
    ],
)
def test_a_structure_must_hold_only_deterministic_immutable_values(extension):
    # A structure is hashed as a registry key and compared across processes, so
    # anything that could run user code or pickle back differently is rejected.
    class ForeignStructureDType(SimpleDType, abstract=False):
        __slots__ = ()

        def structure_extension(self) -> tuple[object, ...]:
            return extension

    with pytest.raises(TypeError, match="holds only exact strings"):
        ForeignStructureDType("TestForeignStructure", bits=8, supertype=DType.Floating)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestForeignStructure")


def test_a_structure_cannot_reference_an_unfinalized_descriptor():
    # The graph is acyclic because a descriptor may only name descriptors that
    # were already published. A descriptor whose construction failed is not one
    # of them, so referring to it is rejected instead of expanding forever.
    escaped: list[DTypeCategory] = []

    class EscapingCategory(DTypeCategory, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            super().__init__(name, supertype=DType.Any)
            escaped.append(self)
            raise RuntimeError("constructor failed after its base initialized")

    with pytest.raises(RuntimeError, match="failed after its base initialized"):
        EscapingCategory("TestEscapedCategory")

    with pytest.raises(ValueError, match="is not finalized"):
        SimpleDType("TestUnderEscaped", bits=8, supertype=escaped[0])
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestUnderEscaped")


def _escape_a_rejected_registration(name: str, sink: list[DTypeCategory]) -> None:
    """Construct a category under a taken ``name``, keeping the rejected object."""

    class EscapingCategory(DTypeCategory, abstract=False):
        __slots__ = ()

        def __init__(self, taken: str) -> None:
            super().__init__(taken, supertype=DType.Any)
            sink.append(self)

    with pytest.raises(ValueError, match="already registered"):
        EscapingCategory(name)


def test_a_rejected_registration_cannot_be_referenced_by_a_later_descriptor():
    # The adversarial case for reference eligibility: this object's constructor
    # succeeded, so its structure was computed before its name was found to be
    # taken. Trusting that recorded structure would let a descriptor that does
    # register sit under a parent nothing resolves to.
    escaped: list[DTypeCategory] = []
    _escape_a_rejected_registration("Floating", escaped)
    rejected = escaped[0]

    assert rejected is not DType.Floating
    assert DType.from_name("Floating") is DType.Floating
    with pytest.raises(AttributeError):
        rejected.structure()

    with pytest.raises(ValueError, match="is not finalized"):
        SimpleDType("TestUnderRejected", bits=8, supertype=rejected)
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestUnderRejected")

    # It is equally ineligible as a compound plane and a block-scaled element,
    # and the rejected object never became immutable, so nothing about it can
    # be mistaken for a finalized descriptor.
    simple_rejected: list[SimpleDType] = []

    class EscapingSimple(SimpleDType, abstract=False):
        __slots__ = ()

        def __init__(self, taken: str) -> None:
            super().__init__(taken, bits=32, supertype=DType.Floating)
            simple_rejected.append(self)

    with pytest.raises(ValueError, match="already registered"):
        EscapingSimple("Float32")
    plane = simple_rejected[0]

    with pytest.raises(ValueError, match="is not finalized"):
        PlanarDType("TestPlaneUnderRejected", planes=(plane,))
    with pytest.raises(ValueError, match="is not finalized"):
        BlockScaledDType(
            "TestBlockUnderRejected",
            element=plane,
            levels=(Level(DType.E8M0, 24),),
        )
    with pytest.raises(ValueError, match="is not finalized"):
        BlockScaledDType(
            "TestScaleUnderRejected",
            element=DType.E2M1,
            levels=(Level(plane, 24),),
        )
    for name in (
        "TestPlaneUnderRejected",
        "TestBlockUnderRejected",
        "TestScaleUnderRejected",
    ):
        with pytest.raises(LookupError, match="No StrideWeave dtype named"):
            DType.from_name(name)

    # A rejected object is inert rather than a sealed descriptor: it kept no
    # structure, and the representation it tried to claim is still free.
    recovered = BlockScaledDType(
        "TestRecoveredAfterRejection",
        element=DType.E2M1,
        levels=(Level(DType.E8M0, 24),),
    )
    assert BlockScaledDType.from_name("TestRecoveredAfterRejection") is recovered


def test_a_descriptor_rejected_for_its_structure_keeps_no_finalization_state():
    # The same rule applies when it is the structural key, not the name, that
    # was already claimed: the rejected object is inert, and the representation
    # stays owned by the descriptor that claimed it first.
    losers: list[BlockScaledDType] = []

    class EscapingBlockScaled(BlockScaledDType, abstract=False):
        __slots__ = ()

        def __init__(self, name: str) -> None:
            super().__init__(name, element=DType.E2M1, levels=(Level(DType.E8M0, 32),))
            losers.append(self)

    with pytest.raises(ValueError, match="'MXFP4' already describes"):
        EscapingBlockScaled("TestLostStructureRace")
    rejected = losers[0]

    with pytest.raises(AttributeError):
        rejected.structure()
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestLostStructureRace")
    with pytest.raises(ValueError, match="'MXFP4' already describes"):
        EscapingBlockScaled("TestLostStructureRaceAgain")
    assert DType.MXFP4.simple_types == (DType.E2M1, DType.E8M0)


def test_a_failed_structure_commit_leaves_the_name_unclaimed(monkeypatch):
    # The structure is claimed before the name, so a structure commit that fails
    # must leave both free rather than burning the name it was constructed with.
    novel_levels = (Level(DType.E8M0, 112),)

    class FailingStructures(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("structure commit failed")

    monkeypatch.setattr(
        dtype_model, "_STRUCTURES", FailingStructures(dtype_model._STRUCTURES)
    )
    with pytest.raises(RuntimeError, match="structure commit failed"):
        BlockScaledDType(
            "TestFailedStructureCommit", element=DType.E4M3, levels=novel_levels
        )
    monkeypatch.undo()

    recovered = BlockScaledDType(
        "TestFailedStructureCommit", element=DType.E4M3, levels=novel_levels
    )
    assert BlockScaledDType.from_name("TestFailedStructureCommit") is recovered
    assert recovered.levels == novel_levels


def test_registry_keys_are_exact_strings_that_cannot_run_user_code():
    class UnstableName(str):
        def __hash__(self):
            raise RuntimeError("this name must never be hashed by the registry")

    extension = SimpleDType(
        UnstableName("TestExactName"), bits=8, supertype=DType.Floating
    )

    assert type(extension.name) is str
    assert extension.name == "TestExactName"
    assert DType.from_name("TestExactName") is extension

    # Nor can an implementation substitute one after its base normalized it.
    class RenamingDType(SimpleDType, abstract=False):
        __slots__ = ()

        def __init__(self) -> None:
            super().__init__("TestRenamed", bits=8, supertype=DType.Floating)
            self._name = UnstableName("TestRenamed")

    with pytest.raises(TypeError, match="name must be an exact string"):
        RenamingDType()
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestRenamed")


def test_failed_registration_leaves_the_name_registry_untouched():
    with pytest.raises(TypeError, match="must belong to a DTypeCategory"):
        SimpleDType("TestNeverRegistered", bits=8, supertype=DType.Float32)  # type: ignore[arg-type]
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name("TestNeverRegistered")


def _run_concurrently(targets):
    """Run each callable in its own thread, released from one barrier.

    Returns each call's result or the exception it raised, in ``targets`` order.
    """
    barrier = threading.Barrier(len(targets))
    outcomes: list[object] = [None] * len(targets)

    def run(position, target):
        barrier.wait()
        try:
            outcomes[position] = target()
        except Exception as error:  # noqa: BLE001 - reported to the assertions
            outcomes[position] = error

    threads = [
        threading.Thread(target=run, args=(position, target))
        for position, target in enumerate(targets)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    return outcomes


def test_registration_commits_inside_the_shared_registry_lock():
    # Holding the registry lock must suspend a competing registration between
    # its name check and its commit; an unsynchronized registry would let the
    # thread run to completion here.
    committed = threading.Event()

    def register():
        SimpleDType("TestLockedRegistration", bits=8, supertype=DType.Floating)
        committed.set()

    with dtype_model._REGISTRY_LOCK:
        thread = threading.Thread(target=register)
        thread.start()
        assert not committed.wait(timeout=0.25)
        with pytest.raises(LookupError, match="No StrideWeave dtype named"):
            DType.from_name("TestLockedRegistration")

    thread.join(timeout=10)
    assert not thread.is_alive()
    assert committed.is_set()
    assert SimpleDType.from_name("TestLockedRegistration").bits == 8


def test_concurrent_same_name_registration_yields_exactly_one_descriptor():
    def construct():
        return SimpleDType("TestRacedName", bits=16, supertype=DType.Floating)

    outcomes = _run_concurrently([construct, construct])
    registered = [item for item in outcomes if isinstance(item, SimpleDType)]
    rejected = [item for item in outcomes if isinstance(item, ValueError)]

    assert len(registered) == 1
    assert len(rejected) == 1
    assert "already registered" in str(rejected[0])
    # Every descriptor a constructor handed back is the registered identity.
    assert SimpleDType.from_name("TestRacedName") is registered[0]
    assert registered[0] in SimpleDType.registered()


def test_concurrent_same_structure_registration_claims_one_structure():
    levels = (Level(DType.E8M0, 128),)

    def construct(name):
        return lambda: BlockScaledDType(
            "TestRaced" + name, element=DType.E4M3, levels=levels
        )

    outcomes = _run_concurrently([construct("StructureA"), construct("StructureB")])
    registered = [item for item in outcomes if isinstance(item, BlockScaledDType)]
    rejected = [item for item in outcomes if isinstance(item, ValueError)]

    assert len(registered) == 1
    assert len(rejected) == 1
    assert "already describes this element and level chain" in str(rejected[0])
    assert BlockScaledDType.from_name(registered[0].name) is registered[0]
    # The losing name was never claimed, so only one identity describes the
    # structure and the rejected name stays free.
    losing_names = {"TestRacedStructureA", "TestRacedStructureB"} - {registered[0].name}
    with pytest.raises(LookupError, match="No StrideWeave dtype named"):
        DType.from_name(losing_names.pop())


def test_whole_extent_is_a_true_singleton():
    assert WholeExtent() is Whole
    assert WholeExtent() is WholeExtent()
    assert copy.copy(Whole) is Whole
    assert copy.deepcopy(Whole) is Whole
    assert pickle.loads(pickle.dumps(Whole)) is Whole


def test_independently_created_whole_levels_cannot_bypass_uniqueness():
    assert Level(DType.Float32, WholeExtent()) == Level(DType.Float32, Whole)
    assert Level(DType.Float32, WholeExtent()).is_whole()

    with pytest.raises(ValueError, match="'NVFP4' already describes"):
        BlockScaledDType(
            "TestNVFP4Clone",
            element=DType.E2M1,
            levels=(
                Level(DType.E4M3, 16),
                Level(DType.Float32, WholeExtent()),
            ),
        )


@pytest.mark.parametrize("name", ["Float32", "Floating", "Int32", "MXFP4", "NVFP4"])
def test_built_in_bindings_reject_reassignment_and_deletion(name):
    original = getattr(DType, name)

    with pytest.raises(AttributeError, match="cannot be reassigned"):
        setattr(DType, name, DType.Any)
    with pytest.raises(AttributeError, match="cannot be deleted"):
        delattr(DType, name)

    assert getattr(DType, name) is original
    assert DType.from_name(name) is original


def test_installing_a_built_in_binding_twice_is_rejected():
    with pytest.raises(AttributeError, match="already installed"):
        DType._install_builtin("Float32", DType.Any)
    assert DType.Float32 is DType.from_name("Float32")


def test_the_built_in_namespace_admits_no_further_descriptor():
    # An extension is registry-only, so binding one as a new class attribute
    # would make the namespace disagree with the documented built-in surface.
    extension = SimpleDType("TestNamespaceFrozen", bits=16, supertype=DType.Floating)

    with pytest.raises(AttributeError, match="namespace holds exactly"):
        DType.TestNamespaceFrozen = extension  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="namespace holds exactly"):
        SimpleDType.TestNamespaceFrozen = extension  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="namespace is frozen"):
        DType._install_builtin("TestNamespaceFrozen", extension)

    assert not hasattr(DType, "TestNamespaceFrozen")
    assert not hasattr(SimpleDType, "TestNamespaceFrozen")
    assert DType.from_name("TestNamespaceFrozen") is extension


def test_namespace_protection_cannot_be_disabled_by_class_assignment():
    # The installed-name bookkeeping lives in module state, so no assignment
    # through the class can replace it and unlock the built-in bindings.
    DType._installed = frozenset()  # type: ignore[attr-defined]
    try:
        with pytest.raises(AttributeError, match="cannot be reassigned"):
            DType.Float32 = DType.Int32  # type: ignore[misc]
        with pytest.raises(AttributeError, match="cannot be deleted"):
            del DType.Float32  # type: ignore[misc]
        with pytest.raises(AttributeError, match="namespace holds exactly"):
            DType.TestUnlocked = DType.Int32  # type: ignore[attr-defined]
    finally:
        del DType._installed  # type: ignore[attr-defined]

    assert DType.Float32 is DType.from_name("Float32")


def test_registry_lookups_keep_their_subclass_type_without_a_cast():
    # These attribute accesses only type-check because registered() and
    # from_name() are annotated with Self rather than the DType root.
    assert SimpleDType.from_name("E4M3").bits == 8
    assert all(dtype.bits > 0 for dtype in SimpleDType.registered())
    assert BlockScaledDType.from_name("MXFP4").num_carriers == 2
    assert all(
        dtype.simple_types[0].bits > 0 for dtype in BlockScaledDType.registered()
    )
    assert all(
        dtype.is_opaque_storage() in (True, False)
        for dtype in DTypeCategory.registered()
    )
