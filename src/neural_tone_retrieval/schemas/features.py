"""Feature artifact records for baseline and learned representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from neural_tone_retrieval.schemas.dataset import SplitName
from neural_tone_retrieval.utils import (
    JsonValue,
    RecordMixin,
    ensure_aware_datetime,
    normalize_optional_string,
    require_non_empty,
    stable_id,
    utc_now,
)


@dataclass(slots=True, frozen=True)
class FeatureRecord(RecordMixin):
    artifact_id: str
    subject_artifact_id: str
    extractor_id: str
    feature_id: str = ""
    feature_set: str = "baseline_handcrafted"
    feature_count: int = 0
    split: SplitName | None = None
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", require_non_empty(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self,
            "subject_artifact_id",
            require_non_empty(self.subject_artifact_id, "subject_artifact_id"),
        )
        object.__setattr__(
            self,
            "extractor_id",
            require_non_empty(self.extractor_id, "extractor_id"),
        )
        object.__setattr__(
            self,
            "feature_set",
            require_non_empty(self.feature_set, "feature_set"),
        )
        if self.feature_count < 0:
            raise ValueError("feature_count must be non-negative")
        ensure_aware_datetime(self.created_at, "created_at")
        feature_id = self.feature_id or stable_id("feature", self.identity_payload())
        object.__setattr__(self, "feature_id", require_non_empty(feature_id, "feature_id"))

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "artifact_id": self.artifact_id,
            "subject_artifact_id": self.subject_artifact_id,
            "extractor_id": self.extractor_id,
            "feature_set": self.feature_set,
        }
