"""Deterministic encoded payloads for kernel verification."""

from __future__ import annotations

import hashlib
import math
import random
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral


def _float32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@dataclass(frozen=True, slots=True)
class EncodedFloat32Payload:
    """Immutable Float32 operand represented by canonical words and a hash."""

    bits: tuple[int, ...]
    bit_hash: str

    @classmethod
    def from_bits(cls, bits: Iterable[int]) -> EncodedFloat32Payload:
        raw_bits = tuple(bits)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_bits
        ):
            raise TypeError("Float32 payload bits must be integral uint32 values")
        encoded = tuple(int(value) for value in raw_bits)
        if any(value < 0 or value > 0xFFFF_FFFF for value in encoded):
            raise ValueError("Float32 payload bits must fit uint32")
        raw = b"".join(struct.pack("<I", value) for value in encoded)
        return cls(encoded, hashlib.sha256(raw).hexdigest())

    @classmethod
    def from_values(cls, values: Iterable[float | int]) -> EncodedFloat32Payload:
        return cls.from_bits(_float32_bits(float(value)) for value in values)

    def values(self) -> tuple[float, ...]:
        return tuple(_float32_from_bits(value) for value in self.bits)


@dataclass(frozen=True, slots=True)
class EncodedInt32Payload:
    """Immutable Int32 operand represented by canonical words and a hash."""

    bits: tuple[int, ...]
    bit_hash: str

    @classmethod
    def from_values(cls, values: Iterable[int]) -> EncodedInt32Payload:
        encoded = []
        for value in values:
            raw = struct.pack("<i", value)
            encoded.append(struct.unpack("<I", raw)[0])
        canonical = b"".join(struct.pack("<I", value) for value in encoded)
        return cls(tuple(encoded), hashlib.sha256(canonical).hexdigest())

    def values(self) -> tuple[int, ...]:
        return tuple(
            struct.unpack("<i", struct.pack("<I", value))[0] for value in self.bits
        )


@dataclass(frozen=True, slots=True)
class EncodedBoolPayload:
    """One immutable Bool operand, encoded as one byte per element.

    Args:
        bits: One ``0`` or ``1`` per element, in logical order.
        bit_hash: SHA-256 of the canonical encoded bytes.
    """

    bits: tuple[int, ...]
    bit_hash: str

    @classmethod
    def from_values(cls, values: Iterable[bool]) -> EncodedBoolPayload:
        """Encode Python bools into a hashed immutable payload.

        Args:
            values: Bool elements in logical order.

        Returns:
            The encoded payload.

        Examples:
            >>> EncodedBoolPayload.from_values((True, False)).bits
            (1, 0)
        """
        encoded = tuple(1 if bool(value) else 0 for value in values)
        raw = bytes(encoded)
        return cls(encoded, hashlib.sha256(raw).hexdigest())

    def values(self) -> tuple[bool, ...]:
        """Return the decoded Bool elements in logical order."""
        return tuple(bool(value) for value in self.bits)


@dataclass(frozen=True, slots=True)
class EncodedInputs:
    """Collection of encoded operands shared by target and oracle execution."""

    operands: tuple[
        EncodedFloat32Payload | EncodedInt32Payload | EncodedBoolPayload, ...
    ]

    @property
    def input_hashes(self) -> tuple[str, ...]:
        return tuple(operand.bit_hash for operand in self.operands)

    def target_values(self) -> tuple[tuple[float | int | bool, ...], ...]:
        return tuple(operand.values() for operand in self.operands)

    def oracle_values(self) -> tuple[tuple[float | int | bool, ...], ...]:
        return tuple(operand.values() for operand in self.operands)


def arbitrary_float32_payload(seed: int, count: int) -> EncodedFloat32Payload:
    """Create deterministic finite Float32 words with varied exponents.

    Args:
        seed: Seed for the deterministic pseudo-random generator.
        count: Number of encoded values to produce.

    Returns:
        Encoded finite Float32 payload.

    Examples:
        >>> len(arbitrary_float32_payload(7, 3).bits)
        3
    """

    if count < 0:
        raise ValueError("payload count must be non-negative")
    generator = random.Random(seed)
    return EncodedFloat32Payload.from_bits(
        (generator.getrandbits(1) << 31)
        | (generator.randint(96, 160) << 23)
        | generator.getrandbits(23)
        for _ in range(count)
    )


def wide_exponent_float32_payload(seed: int, count: int) -> EncodedFloat32Payload:
    """Create deterministic finite Float32 words spanning wide exponents.

    Args:
        seed: Seed for the deterministic pseudo-random generator.
        count: Number of encoded values to produce.

    Returns:
        Encoded finite Float32 payload with varied signs and exponents.

    Examples:
        >>> len(wide_exponent_float32_payload(7, 3).values())
        3
    """
    if count < 0:
        raise ValueError("payload count must be non-negative")
    generator = random.Random(seed)
    bits = []
    for _ in range(count):
        sign = generator.getrandbits(1) << 31
        exponent = generator.randint(1, 254) << 23
        fraction = generator.getrandbits(23)
        bits.append(sign | exponent | fraction)
    return EncodedFloat32Payload.from_bits(bits)


def adversarial_float32_payload() -> EncodedFloat32Payload:
    """Return a fixed payload covering zeros, subnormals, infinities, and NaNs.

    Args:
        None.

    Returns:
        Encoded Float32 payload for bit-preservation checks.

    Examples:
        >>> len(adversarial_float32_payload().bits)
        10
    """
    return EncodedFloat32Payload.from_bits(
        (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x007F_FFFF,
            0x0080_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0001,
            0x7FC1_2345,
        )
    )


@dataclass(frozen=True, slots=True)
class ExactStructuralPayload:
    """Encoded operands whose legal partial results are exact in binary32."""

    lhs: EncodedFloat32Payload
    rhs: EncodedFloat32Payload | None
    contraction_length: int
    operand_bound: int
    mantissa_bits: int


def exact_structural_payload(
    seed: int,
    contraction_length: int,
    *,
    product: bool,
    rows: int = 1,
    mantissa_bits: int = 24,
) -> ExactStructuralPayload:
    """Prepare a deterministic exact structural reduction or product witness.

    Args:
        seed: Seed for deterministic operand generation.
        contraction_length: Number of terms in each reduction fiber.
        product: Whether to generate a pair of operands for multiplication.
        rows: Number of fibers to generate.
        mantissa_bits: Exact-integer mantissa budget used to bound operands.

    Returns:
        Encoded operands and the exactness bounds used to construct them.

    Examples:
        >>> exact_structural_payload(1, 4, product=False).rhs is None
        True
    """
    if contraction_length <= 0 or rows <= 0:
        raise ValueError("contraction length and rows must be positive")
    maximum_exact_integer = 2**mantissa_bits
    term_bound = maximum_exact_integer // contraction_length
    operand_bound = math.isqrt(term_bound) if product else term_bound
    if operand_bound < 1:
        raise ValueError("contraction length exceeds the exact structural range")

    generator = random.Random(seed)
    count = rows * contraction_length
    lhs_values = tuple(
        generator.randint(-operand_bound, operand_bound) for _ in range(count)
    )
    lhs = EncodedFloat32Payload.from_values(lhs_values)
    rhs = None
    if product:
        rhs_values = tuple(
            generator.randint(-operand_bound, operand_bound) for _ in range(count)
        )
        rhs = EncodedFloat32Payload.from_values(rhs_values)
    return ExactStructuralPayload(
        lhs, rhs, contraction_length, operand_bound, mantissa_bits
    )


@dataclass(frozen=True, slots=True)
class AnalyticCase:
    """Independent expected result for a small analytic witness."""

    case_id: str
    operation: str
    inputs: tuple[tuple[float, ...], ...]
    expected: tuple[float, ...]


def analytic_cases() -> tuple[AnalyticCase, ...]:
    """Return the deterministic analytic witnesses used by Stage One.

    Args:
        None.

    Returns:
        Tuple of small reduction and matmul cases with independent results.

    Examples:
        >>> {case.operation for case in analytic_cases()}
        {'matmul', 'reduce_sum'}
    """
    return (
        AnalyticCase(
            "reduce-balanced", "reduce_sum", ((1.0, -1.0, 2.0, -2.0),), (0.0,)
        ),
        AnalyticCase(
            "reduce-geometric", "reduce_sum", ((1.0, 2.0, 4.0, 8.0),), (15.0,)
        ),
        AnalyticCase("matmul-dot", "matmul", ((1.0, 2.0), (3.0, 4.0)), (11.0,)),
        AnalyticCase("matmul-orthogonal", "matmul", ((1.0, 1.0), (1.0, -1.0)), (0.0,)),
        AnalyticCase("matmul-signed", "matmul", ((-2.0, 3.0), (4.0, 5.0)), (7.0,)),
    )
