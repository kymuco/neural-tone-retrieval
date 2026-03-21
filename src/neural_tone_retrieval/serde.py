"""Serialization helpers for manifests and controlled re-amp configs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tomllib

from neural_tone_retrieval.catalog.manifests import DatasetManifest
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
    FeatureRecord,
    FeatureSubjectType,
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


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return dataset_manifest_from_dict(payload)


def save_dataset_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def dataset_manifest_from_dict(payload: dict[str, object]) -> DatasetManifest:
    return DatasetManifest(
        dataset_name=_require_str(payload["dataset_name"], "dataset_name"),
        dataset_version=_require_str(payload["dataset_version"], "dataset_version"),
        artifacts=tuple(artifact_from_dict(item) for item in _require_list(payload, "artifacts")),
        source_clips=tuple(source_clip_from_dict(item) for item in _require_list(payload, "source_clips")),
        chain_specs=tuple(chain_spec_from_dict(item) for item in _require_list(payload, "chain_specs")),
        renders=tuple(render_from_dict(item) for item in _require_list(payload, "renders")),
        features=tuple(feature_from_dict(item) for item in _require_list(payload, "features")),
        embeddings=tuple(
            embedding_from_dict(item) for item in _require_list(payload, "embeddings")
        ),
        split_assignments=tuple(
            split_assignment_from_dict(item)
            for item in _require_list(payload, "split_assignments")
        ),
        runs=tuple(run_from_dict(item) for item in _require_list(payload, "runs")),
        created_at=_parse_datetime(payload["created_at"]),
    )


def load_controlled_reamp_config(path: str | Path) -> ControlledReampConfig:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return controlled_reamp_config_from_dict(payload)


def controlled_reamp_config_from_dict(payload: dict[str, object]) -> ControlledReampConfig:
    dataset_payload = _require_mapping(payload, "dataset")
    render_payload = _require_mapping(payload, "render")
    retrieval_payload = _require_mapping(payload, "retrieval")
    notes = payload.get("notes")
    if isinstance(notes, dict):
        notes = notes.get("description")

    return ControlledReampConfig(
        config_version=_get_str(payload, "config_version", default="v1"),
        dataset=DatasetSection(
            name=_require_str(dataset_payload["name"], "dataset.name"),
            version=_require_str(dataset_payload["version"], "dataset.version"),
            manifest_uri=_require_str(dataset_payload["manifest_uri"], "dataset.manifest_uri"),
            raw_di_root=_require_str(dataset_payload["raw_di_root"], "dataset.raw_di_root"),
            output_root=_require_str(dataset_payload["output_root"], "dataset.output_root"),
            split_protocol_id=_get_str(
                dataset_payload,
                "split_protocol_id",
                default="content_disjoint_v1",
            ),
            selected_splits=tuple(
                SplitName(item) for item in _require_list(dataset_payload, "selected_splits")
            ),
        ),
        render=RenderSection(
            target_sample_rate_hz=_get_int(
                render_payload,
                "target_sample_rate_hz",
                default=44_100,
            ),
            normalize_input=_get_bool(render_payload, "normalize_input", default=True),
            peak_target_dbfs=_get_optional_float(render_payload, "peak_target_dbfs", default=-1.0),
            tail_sec=_get_float(render_payload, "tail_sec", default=0.25),
            include_chain_ids=tuple(
                _require_str(item, "render.include_chain_ids[]")
                for item in _require_list(render_payload, "include_chain_ids")
            ),
        ),
        retrieval=RetrievalSection(
            feature_sets=tuple(
                FeatureSet(item) for item in _require_list(retrieval_payload, "feature_sets")
            ),
            embedding_model_id=_get_str(
                retrieval_payload,
                "embedding_model_id",
                default="baseline-handcrafted-v1",
            ),
            distance_metric=DistanceMetric(
                _get_str(retrieval_payload, "distance_metric", default="cosine")
            ),
            top_k=_get_int(retrieval_payload, "top_k", default=5),
        ),
        notes=_optional_str(notes, "notes"),
    )


def render_controlled_reamp_config_toml(config: ControlledReampConfig) -> str:
    dataset = config.dataset
    render = config.render
    retrieval = config.retrieval
    lines = [
        f'config_version = "{config.config_version}"',
        "",
        "[dataset]",
        f'name = "{dataset.name}"',
        f'version = "{dataset.version}"',
        f'manifest_uri = "{dataset.manifest_uri}"',
        f'raw_di_root = "{dataset.raw_di_root}"',
        f'output_root = "{dataset.output_root}"',
        f'split_protocol_id = "{dataset.split_protocol_id}"',
        f"selected_splits = [{', '.join(_toml_string(split.value) for split in dataset.selected_splits)}]",
        "",
        "[render]",
        f"target_sample_rate_hz = {render.target_sample_rate_hz}",
        f"normalize_input = {_toml_bool(render.normalize_input)}",
        f"peak_target_dbfs = {_toml_number(render.peak_target_dbfs)}",
        f"tail_sec = {_toml_number(render.tail_sec)}",
        f"include_chain_ids = [{', '.join(_toml_string(chain_id) for chain_id in render.include_chain_ids)}]",
        "",
        "[retrieval]",
        f"feature_sets = [{', '.join(_toml_string(item.value) for item in retrieval.feature_sets)}]",
        f'embedding_model_id = "{retrieval.embedding_model_id}"',
        f'distance_metric = "{retrieval.distance_metric.value}"',
        f"top_k = {retrieval.top_k}",
    ]
    if config.notes is not None:
        lines.extend(
            [
                "",
                "[notes]",
                f'description = "{config.notes}"',
            ]
        )
    return "\n".join(lines) + "\n"


def artifact_from_dict(payload: object) -> ArtifactRecord:
    mapping = _as_mapping(payload, "artifact")
    return ArtifactRecord(
        artifact_type=ArtifactType(mapping["artifact_type"]),
        uri=_require_str(mapping["uri"], "artifact.uri"),
        format=ArtifactFormat(mapping["format"]),
        artifact_id=_get_str(mapping, "artifact_id", default=""),
        sha256=_optional_str(mapping.get("sha256"), "artifact.sha256"),
        size_bytes=_get_optional_int(mapping, "size_bytes"),
        created_at=_parse_datetime(mapping["created_at"]),
        run_id=_optional_str(mapping.get("run_id"), "artifact.run_id"),
        dataset_version=_optional_str(mapping.get("dataset_version"), "artifact.dataset_version"),
        parent_artifact_ids=tuple(
            _require_str(item, "artifact.parent_artifact_ids[]")
            for item in _require_list(mapping, "parent_artifact_ids")
        ),
        attrs=_require_mapping(mapping, "attrs"),
    )


def source_clip_from_dict(payload: object) -> SourceClipRecord:
    mapping = _as_mapping(payload, "source_clip")
    return SourceClipRecord(
        artifact_id=_require_str(mapping["artifact_id"], "source_clip.artifact_id"),
        content_group_id=_require_str(
            mapping["content_group_id"],
            "source_clip.content_group_id",
        ),
        source_clip_id=_get_str(mapping, "source_clip_id", default=""),
        session_id=_optional_str(mapping.get("session_id"), "source_clip.session_id"),
        player_id=_optional_str(mapping.get("player_id"), "source_clip.player_id"),
        guitar_id=_optional_str(mapping.get("guitar_id"), "source_clip.guitar_id"),
        pickup_position=_optional_str(mapping.get("pickup_position"), "source_clip.pickup_position"),
        tuning=_optional_str(mapping.get("tuning"), "source_clip.tuning"),
        string_gauge=_optional_str(mapping.get("string_gauge"), "source_clip.string_gauge"),
        technique_tags=tuple(
            _require_str(item, "source_clip.technique_tags[]")
            for item in _require_list(mapping, "technique_tags")
        ),
        bpm=_get_optional_float(mapping, "bpm"),
        key=_optional_str(mapping.get("key"), "source_clip.key"),
        duration_sec=_get_optional_float(mapping, "duration_sec"),
        sample_rate_hz=_get_optional_int(mapping, "sample_rate_hz"),
        channels=_get_optional_int(mapping, "channels"),
        license_ref=_optional_str(mapping.get("license_ref"), "source_clip.license_ref"),
    )


def chain_stage_from_dict(payload: object) -> ChainStage:
    mapping = _as_mapping(payload, "chain_stage")
    return ChainStage(
        stage_index=_get_int(mapping, "stage_index"),
        stage_type=StageType(mapping["stage_type"]),
        processor_id=_require_str(mapping["processor_id"], "chain_stage.processor_id"),
        processor_version=_optional_str(
            mapping.get("processor_version"),
            "chain_stage.processor_version",
        ),
        bypass=_get_bool(mapping, "bypass", default=False),
        params=_require_mapping(mapping, "params"),
    )


def chain_spec_from_dict(payload: object) -> ChainSpec:
    mapping = _as_mapping(payload, "chain_spec")
    return ChainSpec(
        chain_name=_require_str(mapping["chain_name"], "chain_spec.chain_name"),
        chain_family=_require_str(mapping["chain_family"], "chain_spec.chain_family"),
        stages=tuple(chain_stage_from_dict(item) for item in _require_list(mapping, "stages")),
        chain_id=_get_str(mapping, "chain_id", default=""),
        chain_version=_get_str(mapping, "chain_version", default="v1"),
        target_sample_rate_hz=_get_int(mapping, "target_sample_rate_hz", default=44_100),
        amp_family=_optional_str(mapping.get("amp_family"), "chain_spec.amp_family"),
        cab_family=_optional_str(mapping.get("cab_family"), "chain_spec.cab_family"),
        ir_id=_optional_str(mapping.get("ir_id"), "chain_spec.ir_id"),
        gain_bucket=_optional_str(mapping.get("gain_bucket"), "chain_spec.gain_bucket"),
        brightness_bucket=_optional_str(
            mapping.get("brightness_bucket"),
            "chain_spec.brightness_bucket",
        ),
        fx_tags=tuple(
            _require_str(item, "chain_spec.fx_tags[]")
            for item in _require_list(mapping, "fx_tags")
        ),
        notes=_optional_str(mapping.get("notes"), "chain_spec.notes"),
    )


def render_from_dict(payload: object) -> RenderRecord:
    mapping = _as_mapping(payload, "render")
    split = mapping.get("split")
    return RenderRecord(
        artifact_id=_require_str(mapping["artifact_id"], "render.artifact_id"),
        source_clip_id=_require_str(mapping["source_clip_id"], "render.source_clip_id"),
        chain_id=_require_str(mapping["chain_id"], "render.chain_id"),
        render_id=_get_str(mapping, "render_id", default=""),
        render_config_hash=_optional_str(
            mapping.get("render_config_hash"),
            "render.render_config_hash",
        ),
        split=SplitName(split) if split is not None else None,
        duration_sec=_get_optional_float(mapping, "duration_sec"),
        sample_rate_hz=_get_optional_int(mapping, "sample_rate_hz"),
        peak_dbfs=_get_optional_float(mapping, "peak_dbfs"),
        rms_dbfs=_get_optional_float(mapping, "rms_dbfs"),
        integrated_lufs=_get_optional_float(mapping, "integrated_lufs"),
        spectral_centroid_mean=_get_optional_float(mapping, "spectral_centroid_mean"),
        latency_ms=_get_optional_float(mapping, "latency_ms"),
    )


def embedding_from_dict(payload: object) -> EmbeddingRecord:
    mapping = _as_mapping(payload, "embedding")
    split = mapping.get("split")
    return EmbeddingRecord(
        artifact_id=_require_str(mapping["artifact_id"], "embedding.artifact_id"),
        subject_artifact_id=_require_str(
            mapping["subject_artifact_id"],
            "embedding.subject_artifact_id",
        ),
        model_id=_require_str(mapping["model_id"], "embedding.model_id"),
        embedding_dim=_get_int(mapping, "embedding_dim"),
        embedding_id=_get_str(mapping, "embedding_id", default=""),
        checkpoint_id=_optional_str(mapping.get("checkpoint_id"), "embedding.checkpoint_id"),
        normalized=_get_bool(mapping, "normalized", default=True),
        split=SplitName(split) if split is not None else None,
        created_at=_parse_datetime(mapping["created_at"]),
    )


def feature_from_dict(payload: object) -> FeatureRecord:
    mapping = _as_mapping(payload, "feature")
    split = mapping.get("split")
    return FeatureRecord(
        artifact_id=_require_str(mapping["artifact_id"], "feature.artifact_id"),
        subject_artifact_id=_require_str(
            mapping["subject_artifact_id"],
            "feature.subject_artifact_id",
        ),
        extractor_id=_require_str(mapping["extractor_id"], "feature.extractor_id"),
        feature_id=_get_str(mapping, "feature_id", default=""),
        subject_type=FeatureSubjectType(
            _get_str(mapping, "subject_type", default=FeatureSubjectType.SOURCE_CLIP.value)
        ),
        feature_set=_get_str(mapping, "feature_set", default="baseline_handcrafted"),
        feature_count=_get_int(mapping, "feature_count", default=0),
        split=SplitName(split) if split is not None else None,
        created_at=_parse_datetime(mapping["created_at"]),
    )


def split_assignment_from_dict(payload: object) -> SplitAssignment:
    mapping = _as_mapping(payload, "split_assignment")
    return SplitAssignment(
        split_protocol_id=_require_str(
            mapping["split_protocol_id"],
            "split_assignment.split_protocol_id",
        ),
        split_protocol_name=_optional_str(
            mapping.get("split_protocol_name"),
            "split_assignment.split_protocol_name",
        ),
        group_type=SplitGroupType(mapping["group_type"]),
        group_id=_require_str(mapping["group_id"], "split_assignment.group_id"),
        split=SplitName(mapping["split"]),
    )


def search_query_from_dict(payload: object) -> SearchQuery:
    mapping = _as_mapping(payload, "search_query")
    return SearchQuery(
        query_type=QueryType(mapping["query_type"]),
        model_id=_require_str(mapping["model_id"], "search_query.model_id"),
        query_id=_get_str(mapping, "query_id", default=""),
        query_artifact_id=_optional_str(mapping.get("query_artifact_id"), "search_query.query_artifact_id"),
        external_query_uri=_optional_str(
            mapping.get("external_query_uri"),
            "search_query.external_query_uri",
        ),
        top_k=_get_int(mapping, "top_k", default=5),
        filters=_require_mapping(mapping, "filters"),
        created_at=_parse_datetime(mapping["created_at"]),
    )


def search_hit_from_dict(payload: object) -> SearchHit:
    mapping = _as_mapping(payload, "search_hit")
    return SearchHit(
        query_id=_require_str(mapping["query_id"], "search_hit.query_id"),
        rank=_get_int(mapping, "rank"),
        candidate_artifact_id=_require_str(
            mapping["candidate_artifact_id"],
            "search_hit.candidate_artifact_id",
        ),
        source_clip_id=_require_str(mapping["source_clip_id"], "search_hit.source_clip_id"),
        content_group_id=_require_str(
            mapping["content_group_id"],
            "search_hit.content_group_id",
        ),
        candidate_render_id=_optional_str(
            mapping.get("candidate_render_id"),
            "search_hit.candidate_render_id",
        ),
        chain_id=_optional_str(mapping.get("chain_id"), "search_hit.chain_id"),
        score=_get_optional_float(mapping, "score"),
        distance=_get_optional_float(mapping, "distance"),
        amp_family=_optional_str(mapping.get("amp_family"), "search_hit.amp_family"),
        cab_family=_optional_str(mapping.get("cab_family"), "search_hit.cab_family"),
        gain_bucket=_optional_str(mapping.get("gain_bucket"), "search_hit.gain_bucket"),
        preview_uri=_optional_str(mapping.get("preview_uri"), "search_hit.preview_uri"),
        same_content_group=_get_optional_bool(mapping, "same_content_group"),
    )


def run_from_dict(payload: object) -> RunRecord:
    mapping = _as_mapping(payload, "run")
    finished_at = mapping.get("finished_at")
    return RunRecord(
        run_type=RunType(mapping["run_type"]),
        run_id=_get_str(mapping, "run_id", default=""),
        config_artifact_id=_optional_str(mapping.get("config_artifact_id"), "run.config_artifact_id"),
        code_commit=_optional_str(mapping.get("code_commit"), "run.code_commit"),
        started_at=_parse_datetime(mapping["started_at"]),
        finished_at=_parse_datetime(finished_at) if finished_at is not None else None,
        status=RunStatus(mapping["status"]),
        inputs_json=_require_mapping(mapping, "inputs_json"),
        outputs_json=_require_mapping(mapping, "outputs_json"),
        metrics_json=_require_mapping(mapping, "metrics_json"),
    )


def _require_mapping(payload: dict[str, object], field_name: str) -> dict[str, object]:
    value = payload.get(field_name, {})
    return _as_mapping(value, field_name)


def _as_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _require_list(payload: dict[str, object], field_name: str) -> list[object]:
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name)


def _get_str(payload: dict[str, object], field_name: str, *, default: str = "") -> str:
    value = payload.get(field_name, default)
    if value == "" and default == "":
        return ""
    return _require_str(value, field_name)


def _get_int(payload: dict[str, object], field_name: str, *, default: int | None = None) -> int:
    value = payload.get(field_name, default)
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _get_optional_int(payload: dict[str, object], field_name: str) -> int | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer or null")
    return value


def _get_float(payload: dict[str, object], field_name: str, *, default: float | None = None) -> float:
    value = payload.get(field_name, default)
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise TypeError(f"{field_name} must be a float")
    return value


def _get_optional_float(
    payload: dict[str, object],
    field_name: str,
    *,
    default: float | None = None,
) -> float | None:
    value = payload.get(field_name, default)
    if value is None:
        return None
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise TypeError(f"{field_name} must be a float or null")
    return value


def _get_bool(payload: dict[str, object], field_name: str, *, default: bool) -> bool:
    value = payload.get(field_name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _get_optional_bool(payload: dict[str, object], field_name: str) -> bool | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean or null")
    return value


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("datetime fields must be ISO8601 strings")
    return datetime.fromisoformat(value)


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _toml_number(value: float | None) -> str:
    if value is None:
        return "nan"
    if value == int(value):
        return str(int(value))
    return str(value)
