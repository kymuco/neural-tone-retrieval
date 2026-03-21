"""Pipeline configuration models for controlled re-amping and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from neural_tone_retrieval.schemas.retrieval import DistanceMetric
from neural_tone_retrieval.schemas.dataset import SplitName
from neural_tone_retrieval.settings import (
    DEFAULT_SAMPLE_RATE_HZ,
    DEFAULT_SPLIT_PROTOCOL_ID,
    DEFAULT_TOP_K,
)
from neural_tone_retrieval.utils import RecordMixin, normalize_optional_string, normalize_string_tuple, require_non_empty


class FeatureSet(StrEnum):
    MEL = "mel"
    MFCC = "mfcc"
    SPECTRAL = "spectral"


@dataclass(slots=True, frozen=True)
class DatasetSection(RecordMixin):
    name: str
    version: str
    manifest_uri: str
    raw_di_root: str
    output_root: str
    split_protocol_id: str = DEFAULT_SPLIT_PROTOCOL_ID
    selected_splits: tuple[SplitName, ...] = (
        SplitName.TRAIN,
        SplitName.VAL,
        SplitName.TEST,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_non_empty(self.name, "name"))
        object.__setattr__(self, "version", require_non_empty(self.version, "version"))
        object.__setattr__(
            self,
            "manifest_uri",
            require_non_empty(self.manifest_uri, "manifest_uri"),
        )
        object.__setattr__(self, "raw_di_root", require_non_empty(self.raw_di_root, "raw_di_root"))
        object.__setattr__(self, "output_root", require_non_empty(self.output_root, "output_root"))
        object.__setattr__(
            self,
            "split_protocol_id",
            require_non_empty(self.split_protocol_id, "split_protocol_id"),
        )
        splits = tuple(self.selected_splits)
        if not splits:
            raise ValueError("selected_splits must contain at least one split")
        object.__setattr__(self, "selected_splits", splits)


@dataclass(slots=True, frozen=True)
class RenderSection(RecordMixin):
    target_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    normalize_input: bool = True
    peak_target_dbfs: float | None = -1.0
    tail_sec: float = 0.25
    include_chain_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        if self.tail_sec < 0:
            raise ValueError("tail_sec must be non-negative")
        object.__setattr__(
            self,
            "include_chain_ids",
            normalize_string_tuple(self.include_chain_ids),
        )


@dataclass(slots=True, frozen=True)
class RetrievalSection(RecordMixin):
    feature_sets: tuple[FeatureSet, ...] = (
        FeatureSet.MEL,
        FeatureSet.MFCC,
        FeatureSet.SPECTRAL,
    )
    embedding_model_id: str = "baseline-handcrafted-v1"
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    top_k: int = DEFAULT_TOP_K

    def __post_init__(self) -> None:
        feature_sets = tuple(self.feature_sets)
        if not feature_sets:
            raise ValueError("feature_sets must contain at least one feature set")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        object.__setattr__(self, "feature_sets", feature_sets)
        object.__setattr__(
            self,
            "embedding_model_id",
            require_non_empty(self.embedding_model_id, "embedding_model_id"),
        )


@dataclass(slots=True, frozen=True)
class ControlledReampConfig(RecordMixin):
    dataset: DatasetSection
    render: RenderSection = field(default_factory=RenderSection)
    retrieval: RetrievalSection = field(default_factory=RetrievalSection)
    config_version: str = "v1"
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config_version",
            require_non_empty(self.config_version, "config_version"),
        )
        object.__setattr__(self, "notes", normalize_optional_string(self.notes, "notes"))
