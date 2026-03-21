"""Baseline feature index build and nearest-neighbor search."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from neural_tone_retrieval.catalog.manifests import DatasetManifest
from neural_tone_retrieval.feature_extraction import extract_baseline_wav_features
from neural_tone_retrieval.schemas import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    DistanceMetric,
    QueryType,
    RunRecord,
    RunStatus,
    RunType,
    SearchHit,
    SearchQuery,
)
from neural_tone_retrieval.serde import save_dataset_manifest
from neural_tone_retrieval.utils import require_non_empty

DEFAULT_INDEX_FEATURE_KEYS = (
    "peak_abs",
    "rms",
    "mean_abs",
    "std",
    "zero_crossing_rate",
    "crest_factor",
    "crest_factor_db",
    "spectral_centroid_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
)
BASELINE_INDEX_TYPE = "baseline_feature_index_v1"


@dataclass(slots=True, frozen=True)
class BaselineIndexItem:
    feature_id: str
    feature_artifact_id: str
    subject_artifact_id: str
    source_clip_id: str
    content_group_id: str
    preview_uri: str
    split: str | None
    vector: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class BaselineFeatureIndex:
    index_type: str
    extractor_id: str
    distance_metric: DistanceMetric
    feature_keys: tuple[str, ...]
    feature_stats: dict[str, dict[str, float]]
    items: tuple[BaselineIndexItem, ...]


def build_baseline_index(
    manifest: DatasetManifest,
    *,
    manifest_root: str | Path,
    output_manifest_path: str | Path,
    extractor_id: str = "baseline-handcrafted-v1",
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
    index_dir_name: str = "indices",
) -> DatasetManifest:
    manifest_root_path = Path(manifest_root)
    output_manifest = Path(output_manifest_path)
    output_root = output_manifest.parent
    index_dir = output_root / index_dir_name
    index_dir.mkdir(parents=True, exist_ok=True)

    artifact_index = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    source_by_artifact_id = {source.artifact_id: source for source in manifest.source_clips}
    selected_features = [feature for feature in manifest.features if feature.extractor_id == extractor_id]
    if not selected_features:
        raise ValueError(f"No feature records found for extractor_id={extractor_id!r}")

    feature_payloads: list[tuple[object, dict[str, float | int]]] = []
    for feature in selected_features:
        feature_artifact = artifact_index.get(feature.artifact_id)
        if feature_artifact is None:
            raise KeyError(f"Unknown feature artifact_id: {feature.artifact_id}")
        payload = load_feature_payload(manifest_root_path / Path(feature_artifact.uri))
        feature_payloads.append((feature, payload["features"]))

    feature_keys = select_feature_keys(feature_payloads)
    feature_stats = compute_feature_stats(
        [feature_map for _, feature_map in feature_payloads],
        feature_keys=feature_keys,
    )

    index_items: list[BaselineIndexItem] = []
    for feature, feature_map in feature_payloads:
        source_clip = source_by_artifact_id.get(feature.subject_artifact_id)
        if source_clip is None:
            raise KeyError(f"No source clip found for subject_artifact_id={feature.subject_artifact_id}")
        source_artifact = artifact_index.get(source_clip.artifact_id)
        if source_artifact is None:
            raise KeyError(f"Unknown source artifact_id={source_clip.artifact_id}")
        vector = standardize_feature_vector(feature_map, feature_keys=feature_keys, feature_stats=feature_stats)
        index_items.append(
            BaselineIndexItem(
                feature_id=feature.feature_id,
                feature_artifact_id=feature.artifact_id,
                subject_artifact_id=feature.subject_artifact_id,
                source_clip_id=source_clip.source_clip_id,
                content_group_id=source_clip.content_group_id,
                preview_uri=source_artifact.uri,
                split=feature.split.value if feature.split is not None else None,
                vector=tuple(vector),
            )
        )

    index = BaselineFeatureIndex(
        index_type=BASELINE_INDEX_TYPE,
        extractor_id=extractor_id,
        distance_metric=distance_metric,
        feature_keys=feature_keys,
        feature_stats=feature_stats,
        items=tuple(index_items),
    )
    index_path = index_dir / f"baseline_index__{_sanitize_name(extractor_id)}__{distance_metric.value}.json"
    save_baseline_index(index, index_path)
    index_uri = index_path.relative_to(output_root).as_posix()

    index_artifact = ArtifactRecord(
        artifact_type=ArtifactType.INDEX,
        uri=index_uri,
        format=ArtifactFormat.JSON,
        size_bytes=index_path.stat().st_size,
        parent_artifact_ids=tuple(feature.artifact_id for feature in selected_features),
        attrs={
            "index_type": BASELINE_INDEX_TYPE,
            "extractor_id": extractor_id,
            "distance_metric": distance_metric.value,
            "feature_keys": list(feature_keys),
            "item_count": len(index_items),
        },
    )
    run = RunRecord(
        run_type=RunType.INDEX,
        status=RunStatus.COMPLETED,
        inputs_json={
            "manifest_root": str(manifest_root_path),
            "extractor_id": extractor_id,
            "distance_metric": distance_metric.value,
        },
        outputs_json={
            "index_artifact_id": index_artifact.artifact_id,
            "item_count": len(index_items),
            "output_manifest_path": str(output_manifest),
        },
        metrics_json={},
    )
    augmented = DatasetManifest(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        artifacts=tuple((*manifest.artifacts, index_artifact)),
        source_clips=manifest.source_clips,
        chain_specs=manifest.chain_specs,
        renders=manifest.renders,
        features=manifest.features,
        embeddings=manifest.embeddings,
        split_assignments=manifest.split_assignments,
        runs=tuple((*manifest.runs, run)),
    )
    save_dataset_manifest(augmented, output_manifest)
    return augmented


def search_baseline_index(
    manifest: DatasetManifest,
    *,
    manifest_root: str | Path,
    query_audio_path: str | Path,
    top_k: int = 5,
    index_artifact_id: str | None = None,
) -> tuple[SearchQuery, tuple[SearchHit, ...]]:
    manifest_root_path = Path(manifest_root)
    index_artifact = resolve_index_artifact(manifest, index_artifact_id=index_artifact_id)
    index = load_baseline_index(manifest_root_path / Path(index_artifact.uri))

    query_features = extract_baseline_wav_features(query_audio_path)
    query_vector = standardize_feature_vector(
        query_features,
        feature_keys=index.feature_keys,
        feature_stats=index.feature_stats,
    )
    query = SearchQuery(
        query_type=QueryType.EXTERNAL_AUDIO,
        model_id=index.extractor_id,
        external_query_uri=str(query_audio_path),
        top_k=top_k,
        filters={"distance_metric": index.distance_metric.value},
    )

    scored_hits: list[tuple[float, float, BaselineIndexItem]] = []
    for item in index.items:
        distance, score = measure_distance(
            query_vector,
            item.vector,
            distance_metric=index.distance_metric,
        )
        scored_hits.append((distance, score, item))

    scored_hits.sort(key=lambda item: item[0])
    hits: list[SearchHit] = []
    for rank, (distance, score, item) in enumerate(scored_hits[:top_k], start=1):
        hits.append(
            SearchHit(
                query_id=query.query_id,
                rank=rank,
                candidate_render_id=None,
                candidate_artifact_id=item.subject_artifact_id,
                source_clip_id=item.source_clip_id,
                content_group_id=item.content_group_id,
                chain_id=None,
                score=round(score, 8),
                distance=round(distance, 8),
                preview_uri=item.preview_uri,
                same_content_group=None,
            )
        )
    return query, tuple(hits)


def resolve_index_artifact(
    manifest: DatasetManifest,
    *,
    index_artifact_id: str | None = None,
) -> ArtifactRecord:
    index_artifacts = [
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_type == ArtifactType.INDEX
        and artifact.attrs.get("index_type") == BASELINE_INDEX_TYPE
    ]
    if index_artifact_id is not None:
        for artifact in index_artifacts:
            if artifact.artifact_id == index_artifact_id:
                return artifact
        raise KeyError(f"Unknown baseline index artifact_id={index_artifact_id}")
    if not index_artifacts:
        raise ValueError("No baseline index artifact found in manifest")
    return index_artifacts[-1]


def save_baseline_index(index: BaselineFeatureIndex, path: str | Path) -> Path:
    index_path = Path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_type": index.index_type,
        "extractor_id": index.extractor_id,
        "distance_metric": index.distance_metric.value,
        "feature_keys": list(index.feature_keys),
        "feature_stats": index.feature_stats,
        "items": [
            {
                "feature_id": item.feature_id,
                "feature_artifact_id": item.feature_artifact_id,
                "subject_artifact_id": item.subject_artifact_id,
                "source_clip_id": item.source_clip_id,
                "content_group_id": item.content_group_id,
                "preview_uri": item.preview_uri,
                "split": item.split,
                "vector": list(item.vector),
            }
            for item in index.items
        ],
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return index_path


def load_baseline_index(path: str | Path) -> BaselineFeatureIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = tuple(
        BaselineIndexItem(
            feature_id=require_non_empty(item["feature_id"], "feature_id"),
            feature_artifact_id=require_non_empty(item["feature_artifact_id"], "feature_artifact_id"),
            subject_artifact_id=require_non_empty(item["subject_artifact_id"], "subject_artifact_id"),
            source_clip_id=require_non_empty(item["source_clip_id"], "source_clip_id"),
            content_group_id=require_non_empty(item["content_group_id"], "content_group_id"),
            preview_uri=require_non_empty(item["preview_uri"], "preview_uri"),
            split=item.get("split"),
            vector=tuple(float(value) for value in item["vector"]),
        )
        for item in payload["items"]
    )
    feature_stats = {
        str(key): {
            "mean": float(stats["mean"]),
            "std": float(stats["std"]),
        }
        for key, stats in payload["feature_stats"].items()
    }
    return BaselineFeatureIndex(
        index_type=require_non_empty(payload["index_type"], "index_type"),
        extractor_id=require_non_empty(payload["extractor_id"], "extractor_id"),
        distance_metric=DistanceMetric(payload["distance_metric"]),
        feature_keys=tuple(str(value) for value in payload["feature_keys"]),
        feature_stats=feature_stats,
        items=items,
    )


def load_feature_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Feature artifact payload must be an object: {path}")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise TypeError(f"Feature artifact payload is missing a 'features' object: {path}")
    return payload


def select_feature_keys(
    feature_payloads: list[tuple[object, dict[str, float | int]]],
) -> tuple[str, ...]:
    available = set(DEFAULT_INDEX_FEATURE_KEYS)
    for _, feature_map in feature_payloads:
        available &= set(feature_map.keys())
    selected = tuple(key for key in DEFAULT_INDEX_FEATURE_KEYS if key in available)
    if not selected:
        raise ValueError("No compatible feature keys found for index build")
    return selected


def compute_feature_stats(
    feature_maps: list[dict[str, float | int]],
    *,
    feature_keys: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for key in feature_keys:
        values = [float(feature_map[key]) for feature_map in feature_maps]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        stats[key] = {"mean": mean, "std": std if std > 0 else 1.0}
    return stats


def standardize_feature_vector(
    feature_map: dict[str, float | int],
    *,
    feature_keys: tuple[str, ...],
    feature_stats: dict[str, dict[str, float]],
) -> list[float]:
    return [
        (float(feature_map[key]) - feature_stats[key]["mean"]) / feature_stats[key]["std"]
        for key in feature_keys
    ]


def measure_distance(
    left: list[float] | tuple[float, ...],
    right: list[float] | tuple[float, ...],
    *,
    distance_metric: DistanceMetric,
) -> tuple[float, float]:
    if distance_metric == DistanceMetric.L2:
        distance = math.sqrt(sum((l - r) ** 2 for l, r in zip(left, right)))
        return distance, -distance
    if distance_metric == DistanceMetric.DOT:
        score = sum(l * r for l, r in zip(left, right))
        return -score, score

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 and right_norm == 0:
        score = 1.0 if tuple(left) == tuple(right) else 0.0
    elif left_norm == 0 or right_norm == 0:
        score = 0.0
    else:
        score = sum(l * r for l, r in zip(left, right)) / (left_norm * right_norm)
    return 1.0 - score, score


def _sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
