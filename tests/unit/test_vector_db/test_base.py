"""Tests for backend-neutral vector collection and payload boundaries."""

import math

import pytest

from udsc2026.infrastructure.vector_db.base import (
    validate_collection_name,
    validate_query_vector,
    validate_top_k,
    validate_vector_size,
)


@pytest.mark.parametrize("name", ["legal_chunks", "legal-v1", "legal.v1"])
def test_validate_collection_name_accepts_safe_segments(name: str) -> None:
    assert validate_collection_name(f" {name} ") == name


@pytest.mark.parametrize(
    "name",
    ["", " ", ".", "..", "a/b", r"a\b", "\x00", "C:", "legal chunks"],
)
def test_validate_collection_name_rejects_unsafe_segments(name: str) -> None:
    with pytest.raises(ValueError, match="collection_name"):
        validate_collection_name(name)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_validate_positive_vector_integers_reject_invalid_values(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        validate_top_k(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_vector_size(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "vector",
    [[], [math.nan], [math.inf], [True], ["1.0"]],
)
def test_validate_query_vector_rejects_invalid_values(
    vector: list[object],
) -> None:
    with pytest.raises(ValueError):
        validate_query_vector(vector)  # type: ignore[arg-type]


def test_validate_vector_search_values_accepts_finite_input() -> None:
    assert validate_top_k(5) == 5
    assert validate_vector_size(1024) == 1024
    assert validate_query_vector([0.1, -2]) == [0.1, -2]
