"""Run-level records for provenance and experiment tracking."""

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
    require_non_empty,
    stable_id,
    utc_now,
)


class RunType(StrEnum):
    INGEST = "ingest"
    RENDER = "render"
    FEATURES = "features"
    TRAIN = "train"
    INDEX = "index"
    SEARCH = "search"
    EVAL = "eval"


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True, frozen=True)
class RunRecord(RecordMixin):
    run_type: RunType
    run_id: str = ""
    config_artifact_id: str | None = None
    code_commit: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.PLANNED
    inputs_json: dict[str, JsonValue] = field(default_factory=dict)
    outputs_json: dict[str, JsonValue] = field(default_factory=dict)
    metrics_json: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config_artifact_id",
            normalize_optional_string(self.config_artifact_id, "config_artifact_id"),
        )
        object.__setattr__(
            self,
            "code_commit",
            normalize_optional_string(self.code_commit, "code_commit"),
        )
        ensure_aware_datetime(self.started_at, "started_at")
        if self.finished_at is not None:
            ensure_aware_datetime(self.finished_at, "finished_at")
            if self.finished_at < self.started_at:
                raise ValueError("finished_at cannot be earlier than started_at")
        object.__setattr__(self, "inputs_json", normalize_json_mapping(self.inputs_json))
        object.__setattr__(self, "outputs_json", normalize_json_mapping(self.outputs_json))
        object.__setattr__(self, "metrics_json", normalize_json_mapping(self.metrics_json))
        run_id = self.run_id or stable_id("run", self.identity_payload())
        object.__setattr__(self, "run_id", require_non_empty(run_id, "run_id"))

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "run_type": self.run_type,
            "config_artifact_id": self.config_artifact_id,
            "code_commit": self.code_commit,
            "started_at": self.started_at,
        }
