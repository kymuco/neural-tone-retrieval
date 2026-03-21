"""Manifest objects that group catalog records into a portable snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from neural_tone_retrieval.schemas.artifacts import ArtifactRecord
from neural_tone_retrieval.schemas.chains import ChainSpec
from neural_tone_retrieval.schemas.dataset import RenderRecord, SourceClipRecord, SplitAssignment
from neural_tone_retrieval.schemas.features import FeatureRecord
from neural_tone_retrieval.schemas.retrieval import EmbeddingRecord
from neural_tone_retrieval.schemas.runs import RunRecord
from neural_tone_retrieval.utils import RecordMixin, ensure_aware_datetime, require_non_empty, utc_now


@dataclass(slots=True, frozen=True)
class DatasetManifest(RecordMixin):
    dataset_name: str
    dataset_version: str
    artifacts: tuple[ArtifactRecord, ...] = ()
    source_clips: tuple[SourceClipRecord, ...] = ()
    chain_specs: tuple[ChainSpec, ...] = ()
    renders: tuple[RenderRecord, ...] = ()
    features: tuple[FeatureRecord, ...] = ()
    embeddings: tuple[EmbeddingRecord, ...] = ()
    split_assignments: tuple[SplitAssignment, ...] = ()
    runs: tuple[RunRecord, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_name",
            require_non_empty(self.dataset_name, "dataset_name"),
        )
        object.__setattr__(
            self,
            "dataset_version",
            require_non_empty(self.dataset_version, "dataset_version"),
        )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "source_clips", tuple(self.source_clips))
        object.__setattr__(self, "chain_specs", tuple(self.chain_specs))
        object.__setattr__(self, "renders", tuple(self.renders))
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "embeddings", tuple(self.embeddings))
        object.__setattr__(self, "split_assignments", tuple(self.split_assignments))
        object.__setattr__(self, "runs", tuple(self.runs))
        ensure_aware_datetime(self.created_at, "created_at")

    def summary(self) -> dict[str, int]:
        return {
            "artifacts": len(self.artifacts),
            "source_clips": len(self.source_clips),
            "chain_specs": len(self.chain_specs),
            "renders": len(self.renders),
            "features": len(self.features),
            "embeddings": len(self.embeddings),
            "split_assignments": len(self.split_assignments),
            "runs": len(self.runs),
        }
