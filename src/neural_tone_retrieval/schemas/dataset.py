"""Dataset records for source clips, renders, and split assignments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from neural_tone_retrieval.utils import (
    JsonValue,
    RecordMixin,
    normalize_optional_string,
    normalize_string_tuple,
    require_non_empty,
    stable_digest,
    stable_id,
)


class SplitName(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HOLDOUT = "holdout"


class SplitGroupType(StrEnum):
    CONTENT_GROUP = "content_group"
    SESSION = "session"
    GUITAR = "guitar"


@dataclass(slots=True, frozen=True)
class SourceClipRecord(RecordMixin):
    artifact_id: str
    content_group_id: str
    source_clip_id: str = ""
    session_id: str | None = None
    player_id: str | None = None
    guitar_id: str | None = None
    pickup_position: str | None = None
    tuning: str | None = None
    string_gauge: str | None = None
    technique_tags: tuple[str, ...] = ()
    bpm: float | None = None
    key: str | None = None
    duration_sec: float | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    license_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "content_group_id",
            require_non_empty(self.content_group_id, "content_group_id"),
        )
        object.__setattr__(
            self,
            "session_id",
            normalize_optional_string(self.session_id, "session_id"),
        )
        object.__setattr__(
            self,
            "player_id",
            normalize_optional_string(self.player_id, "player_id"),
        )
        object.__setattr__(
            self,
            "guitar_id",
            normalize_optional_string(self.guitar_id, "guitar_id"),
        )
        object.__setattr__(
            self,
            "pickup_position",
            normalize_optional_string(self.pickup_position, "pickup_position"),
        )
        object.__setattr__(self, "tuning", normalize_optional_string(self.tuning, "tuning"))
        object.__setattr__(
            self,
            "string_gauge",
            normalize_optional_string(self.string_gauge, "string_gauge"),
        )
        object.__setattr__(self, "key", normalize_optional_string(self.key, "key"))
        object.__setattr__(
            self,
            "license_ref",
            normalize_optional_string(self.license_ref, "license_ref"),
        )
        object.__setattr__(
            self,
            "technique_tags",
            normalize_string_tuple(self.technique_tags),
        )
        if self.bpm is not None and self.bpm <= 0:
            raise ValueError("bpm must be positive")
        if self.duration_sec is not None and self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels is not None and self.channels <= 0:
            raise ValueError("channels must be positive")
        source_clip_id = self.source_clip_id or stable_id("clip", self.identity_payload())
        object.__setattr__(
            self,
            "source_clip_id",
            require_non_empty(source_clip_id, "source_clip_id"),
        )

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "content_group_id": self.content_group_id,
            "session_id": self.session_id,
            "player_id": self.player_id,
            "guitar_id": self.guitar_id,
            "pickup_position": self.pickup_position,
            "tuning": self.tuning,
            "technique_tags": self.technique_tags,
        }


@dataclass(slots=True, frozen=True)
class RenderRecord(RecordMixin):
    artifact_id: str
    source_clip_id: str
    chain_id: str
    render_id: str = ""
    render_config_hash: str | None = None
    split: SplitName | None = None
    duration_sec: float | None = None
    sample_rate_hz: int | None = None
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    integrated_lufs: float | None = None
    spectral_centroid_mean: float | None = None
    latency_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "source_clip_id",
            require_non_empty(self.source_clip_id, "source_clip_id"),
        )
        object.__setattr__(self, "chain_id", require_non_empty(self.chain_id, "chain_id"))
        if self.duration_sec is not None and self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if self.sample_rate_hz is not None and self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        render_config_hash = self.render_config_hash or stable_digest(
            {
                "artifact_id": self.artifact_id,
                "source_clip_id": self.source_clip_id,
                "chain_id": self.chain_id,
                "sample_rate_hz": self.sample_rate_hz,
            }
        )
        object.__setattr__(self, "render_config_hash", render_config_hash)
        render_id = self.render_id or stable_id("render", self.identity_payload())
        object.__setattr__(self, "render_id", require_non_empty(render_id, "render_id"))

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "source_clip_id": self.source_clip_id,
            "chain_id": self.chain_id,
            "render_config_hash": self.render_config_hash,
        }


@dataclass(slots=True, frozen=True)
class SplitAssignment(RecordMixin):
    split_protocol_id: str
    group_type: SplitGroupType
    group_id: str
    split: SplitName
    split_protocol_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "split_protocol_id",
            require_non_empty(self.split_protocol_id, "split_protocol_id"),
        )
        object.__setattr__(self, "group_id", require_non_empty(self.group_id, "group_id"))
        split_protocol_name = self.split_protocol_name or self.split_protocol_id
        object.__setattr__(
            self,
            "split_protocol_name",
            normalize_optional_string(split_protocol_name, "split_protocol_name"),
        )

    @property
    def key(self) -> tuple[str, SplitGroupType, str]:
        return (self.split_protocol_id, self.group_type, self.group_id)
