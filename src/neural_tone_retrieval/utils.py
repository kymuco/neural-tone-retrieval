"""Shared helpers for stable IDs and record serialization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import TypeAlias, cast

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


class RecordMixin:
    """Small serialization mixin for frozen dataclass records."""

    def to_dict(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], to_json_value(asdict(self)))

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def utc_now() -> datetime:
    return datetime.now(tz=UTC).replace(microsecond=0)


def ensure_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def require_non_empty(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def normalize_optional_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_non_empty(value, field_name)


def normalize_string_tuple(values: Iterable[str]) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in seen:
            items.append(cleaned)
            seen.add(cleaned)
    return tuple(items)


def normalize_json_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    normalized = to_json_value(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError("Expected mapping-like JSON payload")
    return cast(dict[str, JsonValue], normalized)


def stable_digest(value: object, *, length: int = 20) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def stable_id(prefix: str, value: object, *, length: int = 20) -> str:
    return f"{prefix}_{stable_digest(value, length=length)}"


def canonical_json(value: object) -> str:
    return json.dumps(
        to_json_value(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def to_json_value(value: object) -> JsonValue:
    if is_dataclass(value):
        return cast(JsonValue, to_json_value(asdict(value)))
    if isinstance(value, Enum):
        return cast(JsonValue, value.value)
    if isinstance(value, datetime):
        return cast(JsonValue, value.isoformat())
    if isinstance(value, Path):
        return cast(JsonValue, value.as_posix())
    if isinstance(value, Mapping):
        return {
            str(key): to_json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [to_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return cast(JsonValue, value)
    raise TypeError(f"Unsupported value for JSON serialization: {type(value)!r}")
