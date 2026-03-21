"""Retrieval-facing records for embeddings, queries, and hits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from neural_tone_retrieval.settings import DEFAULT_TOP_K
from neural_tone_retrieval.schemas.dataset import SplitName
from neural_tone_retrieval.utils import (
    JsonValue,
    RecordMixin,
    ensure_aware_datetime,
    normalize_json_mapping,
    normalize_optional_string,
    require_non_empty,
    stable_id,
    utc_now,
)


class QueryType(StrEnum):
    RENDERED = "rendered"
    DRY_DI = "dry_di"
    EXTERNAL_AUDIO = "external_audio"


class DistanceMetric(StrEnum):
    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"


@dataclass(slots=True, frozen=True)
class EmbeddingRecord(RecordMixin):
    artifact_id: str
    subject_artifact_id: str
    model_id: str
    embedding_dim: int
    embedding_id: str = ""
    checkpoint_id: str | None = None
    normalized: bool = True
    split: SplitName | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "subject_artifact_id",
            require_non_empty(self.subject_artifact_id, "subject_artifact_id"),
        )
        object.__setattr__(self, "model_id", require_non_empty(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "checkpoint_id",
            normalize_optional_string(self.checkpoint_id, "checkpoint_id"),
        )
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        ensure_aware_datetime(self.created_at, "created_at")
        embedding_id = self.embedding_id or stable_id("embedding", self.identity_payload())
        object.__setattr__(
            self,
            "embedding_id",
            require_non_empty(embedding_id, "embedding_id"),
        )

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "subject_artifact_id": self.subject_artifact_id,
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "embedding_dim": self.embedding_dim,
            "normalized": self.normalized,
        }


@dataclass(slots=True, frozen=True)
class SearchQuery(RecordMixin):
    query_type: QueryType
    model_id: str
    query_id: str = ""
    query_artifact_id: str | None = None
    external_query_uri: str | None = None
    top_k: int = DEFAULT_TOP_K
    filters: dict[str, JsonValue] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", require_non_empty(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "query_artifact_id",
            normalize_optional_string(self.query_artifact_id, "query_artifact_id"),
        )
        object.__setattr__(
            self,
            "external_query_uri",
            normalize_optional_string(self.external_query_uri, "external_query_uri"),
        )
        if bool(self.query_artifact_id) == bool(self.external_query_uri):
            raise ValueError(
                "SearchQuery requires exactly one of query_artifact_id or external_query_uri"
            )
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        object.__setattr__(self, "filters", normalize_json_mapping(self.filters))
        ensure_aware_datetime(self.created_at, "created_at")
        query_id = self.query_id or stable_id("query", self.identity_payload())
        object.__setattr__(self, "query_id", require_non_empty(query_id, "query_id"))

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "query_type": self.query_type,
            "model_id": self.model_id,
            "query_artifact_id": self.query_artifact_id,
            "external_query_uri": self.external_query_uri,
            "top_k": self.top_k,
            "filters": self.filters,
        }


@dataclass(slots=True, frozen=True)
class SearchHit(RecordMixin):
    query_id: str
    rank: int
    candidate_render_id: str
    candidate_artifact_id: str
    source_clip_id: str
    content_group_id: str
    chain_id: str
    score: float | None = None
    distance: float | None = None
    amp_family: str | None = None
    cab_family: str | None = None
    gain_bucket: str | None = None
    preview_uri: str | None = None
    same_content_group: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", require_non_empty(self.query_id, "query_id"))
        object.__setattr__(
            self,
            "candidate_render_id",
            require_non_empty(self.candidate_render_id, "candidate_render_id"),
        )
        object.__setattr__(
            self,
            "candidate_artifact_id",
            require_non_empty(self.candidate_artifact_id, "candidate_artifact_id"),
        )
        object.__setattr__(
            self,
            "source_clip_id",
            require_non_empty(self.source_clip_id, "source_clip_id"),
        )
        object.__setattr__(
            self,
            "content_group_id",
            require_non_empty(self.content_group_id, "content_group_id"),
        )
        object.__setattr__(self, "chain_id", require_non_empty(self.chain_id, "chain_id"))
        object.__setattr__(self, "amp_family", normalize_optional_string(self.amp_family, "amp_family"))
        object.__setattr__(self, "cab_family", normalize_optional_string(self.cab_family, "cab_family"))
        object.__setattr__(
            self,
            "gain_bucket",
            normalize_optional_string(self.gain_bucket, "gain_bucket"),
        )
        object.__setattr__(self, "preview_uri", normalize_optional_string(self.preview_uri, "preview_uri"))
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.score is None and self.distance is None:
            raise ValueError("SearchHit requires at least one of score or distance")
