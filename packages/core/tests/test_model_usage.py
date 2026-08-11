"""Behavioral tests for the ``ModelUsage`` token-usage domain model."""

import pytest
from pydantic import ValidationError

from llmforeman_core import ModelUsage


def test_minimal_construction() -> None:
    usage = ModelUsage(input_tokens=24_322, output_tokens=4_121)
    assert usage.input_tokens == 24_322
    assert usage.output_tokens == 4_121


def test_cache_counters_default_to_zero() -> None:
    usage = ModelUsage(input_tokens=24_322, output_tokens=4_121)
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_all_counters_can_be_zero() -> None:
    usage = ModelUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_cache_counters_can_be_supplied_explicitly() -> None:
    usage = ModelUsage(
        input_tokens=20_000,
        output_tokens=4_000,
        cache_read_input_tokens=8_000,
        cache_creation_input_tokens=2_000,
    )
    assert usage.cache_read_input_tokens == 8_000
    assert usage.cache_creation_input_tokens == 2_000


def test_fully_populated_model_preserves_values() -> None:
    usage = ModelUsage(
        input_tokens=20_000,
        output_tokens=4_000,
        cache_read_input_tokens=8_000,
        cache_creation_input_tokens=2_000,
    )
    assert usage.input_tokens == 20_000
    assert usage.output_tokens == 4_000
    assert usage.cache_read_input_tokens == 8_000
    assert usage.cache_creation_input_tokens == 2_000


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ],
)
def test_negative_counter_rejected(field: str) -> None:
    values = {"input_tokens": 10, "output_tokens": 5}
    values[field] = -1
    with pytest.raises(ValidationError):
        ModelUsage(**values)


@pytest.mark.parametrize("bad_value", [1.5, "not-a-number"])
def test_non_integer_input_rejected(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        ModelUsage.model_validate({"input_tokens": bad_value, "output_tokens": 5})


def test_serialization_preserves_all_counters() -> None:
    usage = ModelUsage(input_tokens=24_322, output_tokens=4_121)
    assert usage.model_dump() == {
        "input_tokens": 24_322,
        "output_tokens": 4_121,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
