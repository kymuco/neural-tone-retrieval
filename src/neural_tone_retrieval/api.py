"""Public Python API for the schema and catalog layer."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Iterable, Sequence

from neural_tone_retrieval.catalog import (
    CatalogRegistry,
    DatasetManifest,
    build_content_split_assignments,
    resolve_render_split,
    resolve_source_clip_split,
)
from neural_tone_retrieval.schemas import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    ChainSpec,
    ChainStage,
    ControlledReampConfig,
    DatasetSection,
    DistanceMetric,
    EmbeddingRecord,
    FeatureSet,
    QueryType,
    RenderRecord,
    RenderSection,
    RetrievalSection,
    RunRecord,
    RunStatus,
    RunType,
    SearchHit,
    SearchQuery,
    SourceClipRecord,
    SplitAssignment,
    SplitGroupType,
    SplitName,
    StageType,
)
from neural_tone_retrieval.serde import (
    load_controlled_reamp_config,
    load_dataset_manifest,
    render_controlled_reamp_config_toml,
    save_dataset_manifest,
)
from neural_tone_retrieval.settings import DEFAULT_DATASET_NAME, DEFAULT_DATASET_VERSION
from neural_tone_retrieval.utils import JsonValue


def create_registry(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    manifest: DatasetManifest | None = None,
) -> CatalogRegistry:
    return CatalogRegistry(
        manifest=manifest,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
    )


def create_dataset_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    artifacts: Iterable[ArtifactRecord] = (),
    source_clips: Iterable[SourceClipRecord] = (),
    chain_specs: Iterable[ChainSpec] = (),
    renders: Iterable[RenderRecord] = (),
    embeddings: Iterable[EmbeddingRecord] = (),
    split_assignments: Iterable[SplitAssignment] = (),
    runs: Iterable[RunRecord] = (),
) -> DatasetManifest:
    return DatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        artifacts=tuple(artifacts),
        source_clips=tuple(source_clips),
        chain_specs=tuple(chain_specs),
        renders=tuple(renders),
        embeddings=tuple(embeddings),
        split_assignments=tuple(split_assignments),
        runs=tuple(runs),
    )


def create_chain_stage(
    *,
    stage_type: StageType,
    processor_id: str,
    params: dict[str, JsonValue] | None = None,
    processor_version: str | None = None,
    bypass: bool = False,
) -> ChainStage:
    return ChainStage(
        stage_index=0,
        stage_type=stage_type,
        processor_id=processor_id,
        processor_version=processor_version,
        bypass=bypass,
        params=params or {},
    )


def create_chain_spec(
    *,
    chain_name: str,
    chain_family: str,
    stages: Sequence[ChainStage],
    chain_version: str = "v1",
    target_sample_rate_hz: int = 44_100,
    amp_family: str | None = None,
    cab_family: str | None = None,
    ir_id: str | None = None,
    gain_bucket: str | None = None,
    brightness_bucket: str | None = None,
    fx_tags: Sequence[str] = (),
    notes: str | None = None,
) -> ChainSpec:
    normalized_stages = tuple(
        replace(stage, stage_index=index)
        for index, stage in enumerate(stages)
    )
    return ChainSpec(
        chain_name=chain_name,
        chain_family=chain_family,
        stages=normalized_stages,
        chain_version=chain_version,
        target_sample_rate_hz=target_sample_rate_hz,
        amp_family=amp_family,
        cab_family=cab_family,
        ir_id=ir_id,
        gain_bucket=gain_bucket,
        brightness_bucket=brightness_bucket,
        fx_tags=tuple(fx_tags),
        notes=notes,
    )


__all__ = [
    "ArtifactFormat",
    "ArtifactRecord",
    "ArtifactType",
    "CatalogRegistry",
    "ChainSpec",
    "ChainStage",
    "ControlledReampConfig",
    "DatasetSection",
    "DatasetManifest",
    "DistanceMetric",
    "EmbeddingRecord",
    "FeatureSet",
    "QueryType",
    "RenderRecord",
    "RenderSection",
    "RetrievalSection",
    "RunRecord",
    "RunStatus",
    "RunType",
    "SearchHit",
    "SearchQuery",
    "SourceClipRecord",
    "SplitAssignment",
    "SplitGroupType",
    "SplitName",
    "StageType",
    "build_content_split_assignments",
    "create_chain_spec",
    "create_chain_stage",
    "create_dataset_manifest",
    "create_registry",
    "load_controlled_reamp_config",
    "load_dataset_manifest",
    "render_controlled_reamp_config_toml",
    "resolve_render_split",
    "resolve_source_clip_split",
    "save_dataset_manifest",
]
