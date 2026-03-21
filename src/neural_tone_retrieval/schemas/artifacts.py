"""Artifact-level records for files produced or tracked by the system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from neural_tone_retrieval.utils import (
    JsonValue,
    RecordMixin,
    ensure_aware_datetime,
    normalize_json_mapping,
    normalize_optional_string,
    normalize_string_tuple,
    require_non_empty,
    stable_id,
    utc_now,
)


class ArtifactType(StrEnum):
    SOURCE_CLIP = "source_clip"
    RENDERED_CLIP = "rendered_clip"
    FEATURES = "features"
    EMBEDDING = "embedding"
    INDEX = "index"
    REPORT = "report"
    CONFIG = "config"


class ArtifactFormat(StrEnum):
    WAV = "wav"
    NPY = "npy"
    PARQUET = "parquet"
    FAISS = "faiss"
    JSON = "json"
    MD = "md"
    YAML = "yaml"
    TOML = "toml"
    BIN = "bin"


@dataclass(slots=True, frozen=True)
class ArtifactRecord(RecordMixin):
    artifact_type: ArtifactType
    uri: str
    format: ArtifactFormat
    artifact_id: str = ""
    sha256: str | None = None
    size_bytes: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    run_id: str | None = None
    dataset_version: str | None = None
    parent_artifact_ids: tuple[str, ...] = ()
    attrs: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", require_non_empty(self.uri, "uri"))
        object.__setattr__(self, "run_id", normalize_optional_string(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "dataset_version",
            normalize_optional_string(self.dataset_version, "dataset_version"),
        )
        object.__setattr__(
            self,
            "parent_artifact_ids",
            normalize_string_tuple(self.parent_artifact_ids),
        )
        object.__setattr__(self, "attrs", normalize_json_mapping(self.attrs))
        ensure_aware_datetime(self.created_at, "created_at")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        artifact_id = self.artifact_id or stable_id("artifact", self.identity_payload())
        object.__setattr__(self, "artifact_id", require_non_empty(artifact_id, "artifact_id"))

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact_type": self.artifact_type,
            "uri": self.uri,
            "format": self.format,
            "sha256": self.sha256,
            "dataset_version": self.dataset_version,
            "run_id": self.run_id,
            "parent_artifact_ids": self.parent_artifact_ids,
        }
