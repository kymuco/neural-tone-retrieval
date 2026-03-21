"""Dataset ingest helpers for scanning dry DI WAV files into a manifest."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import wave

from neural_tone_retrieval.catalog.manifests import DatasetManifest
from neural_tone_retrieval.catalog.splits import build_content_split_assignments
from neural_tone_retrieval.schemas import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    RunRecord,
    RunStatus,
    RunType,
    SourceClipRecord,
    SplitAssignment,
    SplitName,
)
from neural_tone_retrieval.settings import DEFAULT_DATASET_NAME, DEFAULT_DATASET_VERSION
from neural_tone_retrieval.serde import save_dataset_manifest
from neural_tone_retrieval.utils import require_non_empty


@dataclass(slots=True, frozen=True)
class WavFileInfo:
    sample_rate_hz: int
    channels: int
    frame_count: int
    sample_width_bytes: int
    duration_sec: float


def inspect_wav_file(path: str | Path) -> WavFileInfo:
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as stream:
        sample_rate_hz = stream.getframerate()
        channels = stream.getnchannels()
        frame_count = stream.getnframes()
        sample_width_bytes = stream.getsampwidth()
    duration_sec = round(frame_count / sample_rate_hz, 6)
    return WavFileInfo(
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        frame_count=frame_count,
        sample_width_bytes=sample_width_bytes,
        duration_sec=duration_sec,
    )


def ingest_dataset_directory(
    input_dir: str | Path,
    *,
    output_manifest_path: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_version: str = DEFAULT_DATASET_VERSION,
    pattern: str = "*.wav",
    recursive: bool = True,
    compute_sha256: bool = False,
) -> DatasetManifest:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path must be a directory: {root}")

    wav_paths = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    wav_paths = [path for path in wav_paths if path.is_file()]
    if not wav_paths:
        raise FileNotFoundError(f"No audio files matched {pattern!r} under {root}")

    artifacts: list[ArtifactRecord] = []
    source_clips: list[SourceClipRecord] = []
    split_map: dict[str, SplitName] = {}
    total_duration_sec = 0.0

    for wav_path in wav_paths:
        if wav_path.suffix.lower() != ".wav":
            continue
        audio = inspect_wav_file(wav_path)
        sidecar = load_sidecar_metadata(wav_path)
        relative_uri = wav_path.relative_to(root).as_posix()

        artifact = ArtifactRecord(
            artifact_type=ArtifactType.SOURCE_CLIP,
            uri=relative_uri,
            format=ArtifactFormat.WAV,
            sha256=compute_file_sha256(wav_path) if compute_sha256 else None,
            size_bytes=wav_path.stat().st_size,
            attrs={
                "frame_count": audio.frame_count,
                "sample_width_bytes": audio.sample_width_bytes,
            },
        )
        source_clip = SourceClipRecord(
            artifact_id=artifact.artifact_id,
            content_group_id=_get_string(sidecar, "content_group_id", default=wav_path.stem),
            source_clip_id=_get_string(sidecar, "source_clip_id", default=""),
            session_id=_get_optional_string(sidecar, "session_id"),
            player_id=_get_optional_string(sidecar, "player_id"),
            guitar_id=_get_optional_string(sidecar, "guitar_id"),
            pickup_position=_get_optional_string(sidecar, "pickup_position"),
            tuning=_get_optional_string(sidecar, "tuning"),
            string_gauge=_get_optional_string(sidecar, "string_gauge"),
            technique_tags=_get_tags(sidecar),
            bpm=_get_optional_float(sidecar, "bpm"),
            key=_get_optional_string(sidecar, "key"),
            duration_sec=audio.duration_sec,
            sample_rate_hz=audio.sample_rate_hz,
            channels=audio.channels,
            license_ref=_get_optional_string(sidecar, "license_ref"),
        )

        split_name = _get_optional_string(sidecar, "split")
        if split_name is not None:
            split = SplitName(split_name)
            previous_split = split_map.get(source_clip.content_group_id)
            if previous_split is not None and previous_split != split:
                raise ValueError(
                    "Conflicting split assignments for content_group_id "
                    f"{source_clip.content_group_id!r}: {previous_split.value!r} vs {split.value!r}"
                )
            split_map[source_clip.content_group_id] = split

        artifacts.append(artifact)
        source_clips.append(source_clip)
        total_duration_sec += source_clip.duration_sec or 0.0

    split_assignments = tuple(
        _build_split_assignment(content_group_id, split)
        for content_group_id, split in sorted(split_map.items())
    )
    run = RunRecord(
        run_type=RunType.INGEST,
        status=RunStatus.COMPLETED,
        inputs_json={
            "input_dir": str(root),
            "pattern": pattern,
            "recursive": recursive,
            "compute_sha256": compute_sha256,
        },
        outputs_json={
            "source_clips": len(source_clips),
            "artifacts": len(artifacts),
            "split_assignments": len(split_assignments),
        },
        metrics_json={
            "total_duration_sec": round(total_duration_sec, 6),
        },
    )
    manifest = DatasetManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        artifacts=tuple(artifacts),
        source_clips=tuple(source_clips),
        split_assignments=split_assignments,
        runs=(run,),
    )
    if output_manifest_path is not None:
        save_dataset_manifest(manifest, output_manifest_path)
    return manifest


def load_sidecar_metadata(wav_path: str | Path) -> dict[str, object]:
    path = Path(wav_path)
    primary = path.with_suffix(".json")
    secondary = Path(str(path) + ".json")
    candidates = [candidate for candidate in (primary, secondary) if candidate.exists()]
    if len(candidates) > 1:
        raise FileExistsError(
            f"Multiple sidecar files found for {path.name}: {', '.join(str(item.name) for item in candidates)}"
        )
    if not candidates:
        return {}
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Sidecar metadata must be a JSON object: {candidates[0]}")
    return payload


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_split_assignment(content_group_id: str, split: SplitName) -> SplitAssignment:
    return build_content_split_assignments({content_group_id: split})[0]


def _get_string(payload: dict[str, object], key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if value == default and default == "":
        return default
    if not isinstance(value, str):
        raise TypeError(f"Sidecar field {key!r} must be a string")
    return require_non_empty(value, key)


def _get_optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Sidecar field {key!r} must be a string")
    return require_non_empty(value, key)


def _get_optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise TypeError(f"Sidecar field {key!r} must be a number")
    return value


def _get_tags(payload: dict[str, object]) -> tuple[str, ...]:
    value = payload.get("technique_tags", ())
    if isinstance(value, str):
        return (require_non_empty(value, "technique_tags"),)
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError("Sidecar field 'technique_tags' must be a string or list of strings")
    tags: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("Sidecar field 'technique_tags' must contain only strings")
        tags.append(require_non_empty(item, "technique_tags"))
    return tuple(tags)
