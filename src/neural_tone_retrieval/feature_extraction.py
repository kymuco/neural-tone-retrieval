"""Baseline handcrafted feature extraction for WAV audio clips."""

from __future__ import annotations

from array import array
import cmath
import json
import math
from pathlib import Path
import wave

from neural_tone_retrieval.catalog.manifests import DatasetManifest
from neural_tone_retrieval.catalog.splits import resolve_source_clip_split
from neural_tone_retrieval.schemas import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    FeatureRecord,
    FeatureSubjectType,
    RunRecord,
    RunStatus,
    RunType,
)
from neural_tone_retrieval.serde import save_dataset_manifest


def extract_baseline_wav_features(path: str | Path) -> dict[str, float | int]:
    samples, sample_rate_hz = load_wav_as_mono_floats(path)
    if not samples:
        raise ValueError(f"Audio file contains no frames: {path}")

    frame_count = len(samples)
    duration_sec = frame_count / sample_rate_hz
    peak_abs = max(abs(sample) for sample in samples)
    mean = sum(samples) / frame_count
    rms = math.sqrt(sum(sample * sample for sample in samples) / frame_count)
    mean_abs = sum(abs(sample) for sample in samples) / frame_count
    std = math.sqrt(sum((sample - mean) ** 2 for sample in samples) / frame_count)
    zero_crossings = sum(
        1
        for left, right in zip(samples, samples[1:])
        if (left < 0 <= right) or (left > 0 >= right)
    )
    zero_crossing_rate = zero_crossings / max(frame_count - 1, 1)
    crest_factor = peak_abs / rms if rms > 0 else 0.0
    crest_factor_db = 20.0 * math.log10(crest_factor) if crest_factor > 0 else 0.0

    window_stats = [
        _spectral_stats(window, sample_rate_hz)
        for window in iter_analysis_windows(samples)
    ]
    spectral_centroid_hz = _average(item["spectral_centroid_hz"] for item in window_stats)
    spectral_rolloff_hz = _average(item["spectral_rolloff_hz"] for item in window_stats)
    spectral_flatness = _average(item["spectral_flatness"] for item in window_stats)

    return {
        "duration_sec": round(duration_sec, 6),
        "sample_rate_hz": sample_rate_hz,
        "frame_count": frame_count,
        "peak_abs": round(peak_abs, 8),
        "rms": round(rms, 8),
        "mean_abs": round(mean_abs, 8),
        "std": round(std, 8),
        "zero_crossing_rate": round(zero_crossing_rate, 8),
        "crest_factor": round(crest_factor, 8),
        "crest_factor_db": round(crest_factor_db, 8),
        "spectral_centroid_hz": round(spectral_centroid_hz, 8),
        "spectral_rolloff_hz": round(spectral_rolloff_hz, 8),
        "spectral_flatness": round(spectral_flatness, 8),
    }


def build_feature_manifest(
    manifest: DatasetManifest,
    *,
    audio_root: str | Path,
    output_manifest_path: str | Path,
    extractor_id: str = "baseline-handcrafted-v1",
    feature_dir_name: str = "features",
    subject_type: FeatureSubjectType = FeatureSubjectType.SOURCE_CLIP,
) -> DatasetManifest:
    audio_root_path = Path(audio_root)
    output_manifest = Path(output_manifest_path)
    output_root = output_manifest.parent
    feature_dir = output_root / feature_dir_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    artifact_index = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    source_by_source_clip_id = {source.source_clip_id: source for source in manifest.source_clips}
    existing_pairs = {
        (feature.subject_artifact_id, feature.extractor_id)
        for feature in manifest.features
    }

    new_artifacts: list[ArtifactRecord] = []
    new_features: list[FeatureRecord] = []

    for subject in iter_feature_subjects(
        manifest,
        subject_type=subject_type,
        artifact_index=artifact_index,
        source_by_source_clip_id=source_by_source_clip_id,
    ):
        if (subject["subject_artifact_id"], extractor_id) in existing_pairs:
            raise ValueError(
                f"Feature artifact already exists for subject {subject['subject_artifact_id']} "
                f"and extractor {extractor_id}"
            )
        subject_artifact = artifact_index.get(subject["subject_artifact_id"])
        if subject_artifact is None:
            raise KeyError(f"Unknown subject artifact_id {subject['subject_artifact_id']}")
        audio_path = audio_root_path / Path(subject_artifact.uri)
        feature_values = extract_baseline_wav_features(audio_path)
        feature_payload = {
            "extractor_id": extractor_id,
            "subject_artifact_id": subject["subject_artifact_id"],
            "subject_type": subject["subject_type"].value,
            "source_clip_id": subject["source_clip_id"],
            "content_group_id": subject["content_group_id"],
            "chain_id": subject["chain_id"],
            "render_id": subject["render_id"],
            "features": feature_values,
        }
        feature_stem = subject["render_id"] or subject["source_clip_id"]
        feature_filename = f"{feature_stem}__{_sanitize_name(extractor_id)}.json"
        feature_path = feature_dir / feature_filename
        feature_path.write_text(
            json.dumps(feature_payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        feature_uri = feature_path.relative_to(output_root).as_posix()
        feature_artifact = ArtifactRecord(
            artifact_type=ArtifactType.FEATURES,
            uri=feature_uri,
            format=ArtifactFormat.JSON,
            size_bytes=feature_path.stat().st_size,
            parent_artifact_ids=(subject_artifact.artifact_id,),
            attrs={
                "extractor_id": extractor_id,
                "feature_count": len(feature_values),
                "subject_artifact_id": subject_artifact.artifact_id,
                "subject_type": subject["subject_type"].value,
            },
        )
        feature_record = FeatureRecord(
            artifact_id=feature_artifact.artifact_id,
            subject_artifact_id=subject_artifact.artifact_id,
            extractor_id=extractor_id,
            subject_type=subject["subject_type"],
            feature_count=len(feature_values),
            split=subject["split"],
        )
        new_artifacts.append(feature_artifact)
        new_features.append(feature_record)

    run = RunRecord(
        run_type=RunType.FEATURES,
        status=RunStatus.COMPLETED,
        inputs_json={
            "audio_root": str(audio_root_path),
            "extractor_id": extractor_id,
            "subject_type": subject_type.value,
        },
        outputs_json={
            "feature_artifacts": len(new_artifacts),
            "feature_records": len(new_features),
            "output_manifest_path": str(output_manifest),
        },
        metrics_json={},
    )
    augmented = DatasetManifest(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        artifacts=tuple((*manifest.artifacts, *new_artifacts)),
        source_clips=manifest.source_clips,
        chain_specs=manifest.chain_specs,
        renders=manifest.renders,
        features=tuple((*manifest.features, *new_features)),
        embeddings=manifest.embeddings,
        split_assignments=manifest.split_assignments,
        runs=tuple((*manifest.runs, run)),
    )
    save_dataset_manifest(augmented, output_manifest)
    return augmented


def iter_feature_subjects(
    manifest: DatasetManifest,
    *,
    subject_type: FeatureSubjectType,
    artifact_index: dict[str, ArtifactRecord],
    source_by_source_clip_id: dict[str, object],
) -> list[dict[str, object]]:
    subjects: list[dict[str, object]] = []
    if subject_type == FeatureSubjectType.SOURCE_CLIP:
        for source_clip in manifest.source_clips:
            subjects.append(
                {
                    "subject_type": subject_type,
                    "subject_artifact_id": source_clip.artifact_id,
                    "source_clip_id": source_clip.source_clip_id,
                    "content_group_id": source_clip.content_group_id,
                    "chain_id": None,
                    "render_id": None,
                    "split": resolve_source_clip_split(source_clip, manifest.split_assignments),
                }
            )
        return subjects

    if not manifest.renders:
        raise ValueError("Feature extraction for rendered clips requires at least one render record")
    for render in manifest.renders:
        source_clip = source_by_source_clip_id.get(render.source_clip_id)
        if source_clip is None:
            raise KeyError(f"Unknown source_clip_id for render {render.render_id}: {render.source_clip_id}")
        if render.artifact_id not in artifact_index:
            raise KeyError(f"Unknown artifact_id for render {render.render_id}: {render.artifact_id}")
        subjects.append(
            {
                "subject_type": subject_type,
                "subject_artifact_id": render.artifact_id,
                "source_clip_id": render.source_clip_id,
                "content_group_id": source_clip.content_group_id,
                "chain_id": render.chain_id,
                "render_id": render.render_id,
                "split": render.split,
            }
        )
    return subjects


def load_wav_as_mono_floats(path: str | Path) -> tuple[list[float], int]:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate_hz = stream.getframerate()
        frame_count = stream.getnframes()
        raw_frames = stream.readframes(frame_count)

    samples = _decode_pcm_frames(raw_frames, sample_width)
    if channels > 1:
        mono_samples: list[float] = []
        for index in range(0, len(samples), channels):
            frame = samples[index : index + channels]
            mono_samples.append(sum(frame) / len(frame))
        samples = mono_samples
    return samples, sample_rate_hz


def _decode_pcm_frames(raw_frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [((byte - 128) / 128.0) for byte in raw_frames]
    if sample_width == 2:
        values = array("h")
        values.frombytes(raw_frames)
        return [value / 32768.0 for value in values]
    if sample_width == 4:
        values = array("i")
        values.frombytes(raw_frames)
        return [value / 2147483648.0 for value in values]
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def iter_analysis_windows(
    samples: list[float],
    *,
    window_size: int = 512,
    max_windows: int = 3,
) -> list[list[float]]:
    if len(samples) <= window_size:
        return [_apply_hann_window(samples)]

    last_start = len(samples) - window_size
    if max_windows <= 1:
        starts = [0]
    else:
        starts = sorted(
            {
                round(index * last_start / (max_windows - 1))
                for index in range(max_windows)
            }
        )
    return [_apply_hann_window(samples[start : start + window_size]) for start in starts]


def _apply_hann_window(samples: list[float]) -> list[float]:
    sample_count = len(samples)
    if sample_count == 1:
        return samples[:]
    return [
        sample * (0.5 - 0.5 * math.cos((2.0 * math.pi * index) / (sample_count - 1)))
        for index, sample in enumerate(samples)
    ]


def _spectral_stats(window: list[float], sample_rate_hz: int) -> dict[str, float]:
    sample_count = len(window)
    if sample_count < 2:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
        }

    magnitudes: list[float] = []
    frequencies: list[float] = []
    half = sample_count // 2
    for bin_index in range(half + 1):
        total = 0j
        for sample_index, sample in enumerate(window):
            angle = -2.0j * math.pi * bin_index * sample_index / sample_count
            total += sample * cmath.exp(angle)
        magnitudes.append(abs(total))
        frequencies.append(bin_index * sample_rate_hz / sample_count)

    energy_total = sum(magnitudes)
    if energy_total <= 0:
        return {
            "spectral_centroid_hz": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0,
        }

    centroid = sum(freq * mag for freq, mag in zip(frequencies, magnitudes)) / energy_total

    threshold = energy_total * 0.85
    cumulative = 0.0
    rolloff = 0.0
    for frequency, magnitude in zip(frequencies, magnitudes):
        cumulative += magnitude
        if cumulative >= threshold:
            rolloff = frequency
            break

    epsilon = 1e-12
    log_sum = sum(math.log(magnitude + epsilon) for magnitude in magnitudes)
    geometric_mean = math.exp(log_sum / len(magnitudes))
    arithmetic_mean = energy_total / len(magnitudes)
    flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0

    return {
        "spectral_centroid_hz": centroid,
        "spectral_rolloff_hz": rolloff,
        "spectral_flatness": flatness,
    }


def _average(values: list[float] | tuple[float, ...] | object) -> float:
    if not isinstance(values, (list, tuple)):
        values = list(values)
    return sum(values) / len(values)


def _sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
