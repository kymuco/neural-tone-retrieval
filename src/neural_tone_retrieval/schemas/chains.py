"""Signal-chain specifications for controlled re-amping."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from neural_tone_retrieval.settings import DEFAULT_SAMPLE_RATE_HZ
from neural_tone_retrieval.utils import (
    JsonValue,
    RecordMixin,
    normalize_json_mapping,
    normalize_optional_string,
    normalize_string_tuple,
    require_non_empty,
    stable_id,
)


class StageType(StrEnum):
    GATE = "gate"
    OD = "od"
    AMP = "amp"
    CAB = "cab"
    EQ = "eq"
    REVERB = "reverb"
    POST = "post"


@dataclass(slots=True, frozen=True)
class ChainStage(RecordMixin):
    stage_index: int
    stage_type: StageType
    processor_id: str
    processor_version: str | None = None
    bypass: bool = False
    params: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage_index < 0:
            raise ValueError("stage_index must be non-negative")
        object.__setattr__(
            self,
            "processor_id",
            require_non_empty(self.processor_id, "processor_id"),
        )
        object.__setattr__(
            self,
            "processor_version",
            normalize_optional_string(self.processor_version, "processor_version"),
        )
        object.__setattr__(self, "params", normalize_json_mapping(self.params))


@dataclass(slots=True, frozen=True)
class ChainSpec(RecordMixin):
    chain_name: str
    chain_family: str
    stages: tuple[ChainStage, ...]
    chain_id: str = ""
    chain_version: str = "v1"
    target_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    amp_family: str | None = None
    cab_family: str | None = None
    ir_id: str | None = None
    gain_bucket: str | None = None
    brightness_bucket: str | None = None
    fx_tags: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chain_name", require_non_empty(self.chain_name, "chain_name"))
        object.__setattr__(
            self,
            "chain_family",
            require_non_empty(self.chain_family, "chain_family"),
        )
        object.__setattr__(
            self,
            "chain_version",
            require_non_empty(self.chain_version, "chain_version"),
        )
        if self.target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be positive")
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("ChainSpec requires at least one stage")
        for expected_index, stage in enumerate(stages):
            if stage.stage_index != expected_index:
                raise ValueError("Chain stages must have contiguous zero-based indices")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "amp_family", normalize_optional_string(self.amp_family, "amp_family"))
        object.__setattr__(self, "cab_family", normalize_optional_string(self.cab_family, "cab_family"))
        object.__setattr__(self, "ir_id", normalize_optional_string(self.ir_id, "ir_id"))
        object.__setattr__(
            self,
            "gain_bucket",
            normalize_optional_string(self.gain_bucket, "gain_bucket"),
        )
        object.__setattr__(
            self,
            "brightness_bucket",
            normalize_optional_string(self.brightness_bucket, "brightness_bucket"),
        )
        object.__setattr__(self, "notes", normalize_optional_string(self.notes, "notes"))
        object.__setattr__(self, "fx_tags", normalize_string_tuple(self.fx_tags))
        chain_id = self.chain_id or stable_id("chain", self.identity_payload())
        object.__setattr__(self, "chain_id", require_non_empty(chain_id, "chain_id"))

    def identity_payload(self) -> dict[str, JsonValue]:
        return {
            "chain_family": self.chain_family,
            "chain_version": self.chain_version,
            "target_sample_rate_hz": self.target_sample_rate_hz,
            "amp_family": self.amp_family,
            "cab_family": self.cab_family,
            "ir_id": self.ir_id,
            "gain_bucket": self.gain_bucket,
            "brightness_bucket": self.brightness_bucket,
            "fx_tags": self.fx_tags,
            "stages": self.stages,
        }
