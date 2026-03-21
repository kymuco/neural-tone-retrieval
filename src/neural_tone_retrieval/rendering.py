"""Deterministic offline rendering for controlled tone dataset generation."""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import wave

from neural_tone_retrieval.catalog.manifests import DatasetManifest
from neural_tone_retrieval.catalog.splits import resolve_source_clip_split
from neural_tone_retrieval.feature_extraction import load_wav_as_mono_floats
from neural_tone_retrieval.schemas import (
    ArtifactFormat,
    ArtifactRecord,
    ArtifactType,
    ChainSpec,
    RenderRecord,
    RunRecord,
    RunStatus,
    RunType,
    StageType,
)
from neural_tone_retrieval.serde import save_dataset_manifest


def build_render_manifest(
    manifest: DatasetManifest,
    *,
    audio_root: str | Path,
    output_manifest_path: str | Path,
    render_dir_name: str = "renders",
    include_chain_ids: tuple[str, ...] = (),
    peak_target_dbfs: float | None = -1.0,
    tail_sec: float = 0.0,
) -> DatasetManifest:
    if not manifest.chain_specs:
        raise ValueError("Render build requires at least one chain spec in the manifest")

    audio_root_path = Path(audio_root)
    output_manifest = Path(output_manifest_path)
    output_root = output_manifest.parent
    render_dir = output_root / render_dir_name
    render_dir.mkdir(parents=True, exist_ok=True)

    artifact_index = {artifact.artifact_id: artifact for artifact in manifest.artifacts}
    selected_chain_ids = set(include_chain_ids)
    chain_specs = [
        chain
        for chain in manifest.chain_specs
        if not selected_chain_ids or chain.chain_id in selected_chain_ids
    ]
    if not chain_specs:
        raise ValueError("No chain specs matched the requested render selection")

    new_artifacts: list[ArtifactRecord] = []
    new_renders: list[RenderRecord] = []
    total_duration_sec = 0.0

    for source_clip in manifest.source_clips:
        source_artifact = artifact_index.get(source_clip.artifact_id)
        if source_artifact is None:
            raise KeyError(f"Unknown source artifact for clip {source_clip.source_clip_id}")
        source_audio_path = audio_root_path / Path(source_artifact.uri)
        source_samples, source_sample_rate_hz = load_wav_as_mono_floats(source_audio_path)

        for chain_spec in chain_specs:
            if chain_spec.target_sample_rate_hz != source_sample_rate_hz:
                raise ValueError(
                    "MVP renderer requires source sample rate to match chain target sample rate: "
                    f"{source_sample_rate_hz} != {chain_spec.target_sample_rate_hz}"
                )
            rendered_samples = render_samples_through_chain(
                source_samples,
                sample_rate_hz=source_sample_rate_hz,
                chain_spec=chain_spec,
                peak_target_dbfs=peak_target_dbfs,
                tail_sec=tail_sec,
            )
            filename = f"{source_clip.source_clip_id}__{chain_spec.chain_id}.wav"
            render_path = render_dir / filename
            save_mono_wav(render_path, rendered_samples, sample_rate_hz=source_sample_rate_hz)

            duration_sec = len(rendered_samples) / source_sample_rate_hz
            peak_abs = max((abs(sample) for sample in rendered_samples), default=0.0)
            rms = (
                math.sqrt(sum(sample * sample for sample in rendered_samples) / len(rendered_samples))
                if rendered_samples
                else 0.0
            )
            render_artifact = ArtifactRecord(
                artifact_type=ArtifactType.RENDERED_CLIP,
                uri=render_path.relative_to(output_root).as_posix(),
                format=ArtifactFormat.WAV,
                size_bytes=render_path.stat().st_size,
                parent_artifact_ids=(source_artifact.artifact_id,),
                attrs={
                    "source_clip_id": source_clip.source_clip_id,
                    "chain_id": chain_spec.chain_id,
                    "peak_abs": round(peak_abs, 8),
                },
            )
            render_record = RenderRecord(
                artifact_id=render_artifact.artifact_id,
                source_clip_id=source_clip.source_clip_id,
                chain_id=chain_spec.chain_id,
                split=resolve_source_clip_split(source_clip, manifest.split_assignments),
                duration_sec=round(duration_sec, 6),
                sample_rate_hz=source_sample_rate_hz,
                peak_dbfs=linear_to_dbfs(peak_abs),
                rms_dbfs=linear_to_dbfs(rms),
            )
            new_artifacts.append(render_artifact)
            new_renders.append(render_record)
            total_duration_sec += duration_sec

    run = RunRecord(
        run_type=RunType.RENDER,
        status=RunStatus.COMPLETED,
        inputs_json={
            "audio_root": str(audio_root_path),
            "selected_chain_ids": list(include_chain_ids),
            "peak_target_dbfs": peak_target_dbfs,
            "tail_sec": tail_sec,
        },
        outputs_json={
            "rendered_clips": len(new_renders),
            "output_manifest_path": str(output_manifest),
        },
        metrics_json={
            "total_render_duration_sec": round(total_duration_sec, 6),
        },
    )
    augmented = DatasetManifest(
        dataset_name=manifest.dataset_name,
        dataset_version=manifest.dataset_version,
        artifacts=tuple((*manifest.artifacts, *new_artifacts)),
        source_clips=manifest.source_clips,
        chain_specs=manifest.chain_specs,
        renders=tuple((*manifest.renders, *new_renders)),
        features=manifest.features,
        embeddings=manifest.embeddings,
        split_assignments=manifest.split_assignments,
        runs=tuple((*manifest.runs, run)),
    )
    save_dataset_manifest(augmented, output_manifest)
    return augmented


def render_samples_through_chain(
    samples: list[float],
    *,
    sample_rate_hz: int,
    chain_spec: ChainSpec,
    peak_target_dbfs: float | None,
    tail_sec: float,
) -> list[float]:
    rendered = samples[:]
    for stage in chain_spec.stages:
        if stage.bypass:
            continue
        rendered = apply_stage(
            rendered,
            sample_rate_hz=sample_rate_hz,
            stage_type=stage.stage_type,
            params=stage.params,
        )
    if tail_sec > 0:
        rendered = rendered + [0.0] * int(sample_rate_hz * tail_sec)
    if peak_target_dbfs is not None:
        rendered = normalize_peak(rendered, target_dbfs=peak_target_dbfs)
    return clip_samples(rendered)


def apply_stage(
    samples: list[float],
    *,
    sample_rate_hz: int,
    stage_type: StageType,
    params: dict[str, object],
) -> list[float]:
    if stage_type == StageType.GATE:
        threshold_db = get_float(params, "threshold_db", default=-48.0)
        floor_db = get_float(params, "floor_db", default=-80.0)
        return apply_gate(samples, threshold_db=threshold_db, floor_db=floor_db)
    if stage_type == StageType.OD:
        drive = get_float(params, "drive", default=0.2)
        tone = get_float(params, "tone", default=0.5)
        level = get_float(params, "level", default=0.8)
        return apply_overdrive(samples, sample_rate_hz=sample_rate_hz, drive=drive, tone=tone, level=level)
    if stage_type == StageType.AMP:
        gain = get_float(params, "gain", default=0.6)
        bass = get_float(params, "bass", default=0.5)
        mid = get_float(params, "mid", default=0.5)
        treble = get_float(params, "treble", default=0.5)
        presence = get_float(params, "presence", default=0.5)
        return apply_amp(
            samples,
            sample_rate_hz=sample_rate_hz,
            gain=gain,
            bass=bass,
            mid=mid,
            treble=treble,
            presence=presence,
        )
    if stage_type == StageType.CAB:
        ir_id = str(params.get("ir_id", ""))
        return apply_cabinet(samples, sample_rate_hz=sample_rate_hz, ir_id=ir_id)
    if stage_type == StageType.EQ:
        low_cut_hz = get_float(params, "low_cut_hz", default=80.0)
        high_cut_hz = get_float(params, "high_cut_hz", default=10_000.0)
        low_gain = get_float(params, "low_gain", default=0.5)
        mid_gain = get_float(params, "mid_gain", default=0.5)
        high_gain = get_float(params, "high_gain", default=0.5)
        return apply_three_band_eq(
            samples,
            sample_rate_hz=sample_rate_hz,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            low_gain=map_control_to_gain(low_gain),
            mid_gain=map_control_to_gain(mid_gain),
            high_gain=map_control_to_gain(high_gain),
        )
    if stage_type == StageType.REVERB:
        mix = get_float(params, "mix", default=0.1)
        decay_sec = get_float(params, "decay_sec", default=1.5)
        return apply_reverb(samples, sample_rate_hz=sample_rate_hz, mix=mix, decay_sec=decay_sec)
    if stage_type == StageType.POST:
        output_db = get_float(params, "output_db", default=0.0)
        return apply_gain_db(samples, output_db)
    return samples[:]


def apply_gate(samples: list[float], *, threshold_db: float, floor_db: float) -> list[float]:
    threshold = db_to_linear(threshold_db)
    floor = db_to_linear(floor_db)
    result: list[float] = []
    for sample in samples:
        magnitude = abs(sample)
        if magnitude >= threshold:
            result.append(sample)
        else:
            scale = floor if magnitude == 0 else min(1.0, floor / max(magnitude, 1e-9))
            result.append(sample * scale)
    return result


def apply_overdrive(
    samples: list[float],
    *,
    sample_rate_hz: int,
    drive: float,
    tone: float,
    level: float,
) -> list[float]:
    drive_gain = 1.0 + clamp01(drive) * 18.0
    clipped = [math.tanh(sample * drive_gain) / math.tanh(drive_gain) for sample in samples]
    shaped = apply_tone_control(clipped, sample_rate_hz=sample_rate_hz, tone=tone)
    output_gain = 0.25 + clamp01(level) * 1.25
    return clip_samples([sample * output_gain for sample in shaped])


def apply_amp(
    samples: list[float],
    *,
    sample_rate_hz: int,
    gain: float,
    bass: float,
    mid: float,
    treble: float,
    presence: float,
) -> list[float]:
    pre_gain = 1.0 + clamp01(gain) * 40.0
    saturated = [math.tanh(sample * pre_gain) for sample in samples]
    voiced = apply_three_band_eq(
        saturated,
        sample_rate_hz=sample_rate_hz,
        low_cut_hz=70.0,
        high_cut_hz=12_000.0,
        low_gain=map_control_to_gain(bass),
        mid_gain=map_control_to_gain(mid),
        high_gain=map_control_to_gain((treble + presence) / 2.0),
    )
    return clip_samples([math.tanh(sample * 1.4) for sample in voiced])


def apply_cabinet(samples: list[float], *, sample_rate_hz: int, ir_id: str) -> list[float]:
    ir_key = ir_id.lower()
    if "r121" in ir_key:
        low_cut_hz = 75.0
        high_cut_hz = 5_500.0
        high_gain = 0.92
    elif "sm57" in ir_key:
        low_cut_hz = 90.0
        high_cut_hz = 6_800.0
        high_gain = 1.08
    else:
        low_cut_hz = 80.0
        high_cut_hz = 6_200.0
        high_gain = 1.0
    shaped = apply_three_band_eq(
        samples,
        sample_rate_hz=sample_rate_hz,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        low_gain=1.0,
        mid_gain=1.0,
        high_gain=high_gain,
    )
    return clip_samples([sample * 0.9 for sample in shaped])


def apply_three_band_eq(
    samples: list[float],
    *,
    sample_rate_hz: int,
    low_cut_hz: float,
    high_cut_hz: float,
    low_gain: float,
    mid_gain: float,
    high_gain: float,
) -> list[float]:
    low_band = low_pass_filter(samples, sample_rate_hz=sample_rate_hz, cutoff_hz=250.0)
    high_band = high_pass_filter(samples, sample_rate_hz=sample_rate_hz, cutoff_hz=2_500.0)
    mid_band = [
        sample - low - high
        for sample, low, high in zip(samples, low_band, high_band)
    ]
    mixed = [
        low * low_gain + mid * mid_gain + high * high_gain
        for low, mid, high in zip(low_band, mid_band, high_band)
    ]
    filtered = high_pass_filter(mixed, sample_rate_hz=sample_rate_hz, cutoff_hz=low_cut_hz)
    filtered = low_pass_filter(filtered, sample_rate_hz=sample_rate_hz, cutoff_hz=high_cut_hz)
    return filtered


def apply_tone_control(samples: list[float], *, sample_rate_hz: int, tone: float) -> list[float]:
    bright = high_pass_filter(samples, sample_rate_hz=sample_rate_hz, cutoff_hz=1_500.0)
    dark = low_pass_filter(samples, sample_rate_hz=sample_rate_hz, cutoff_hz=2_500.0)
    blend = clamp01(tone)
    return [
        dark_sample * (1.0 - blend) + bright_sample * blend
        for dark_sample, bright_sample in zip(dark, bright)
    ]


def apply_reverb(
    samples: list[float],
    *,
    sample_rate_hz: int,
    mix: float,
    decay_sec: float,
) -> list[float]:
    mix = clamp01(mix)
    delay_samples = max(1, int(sample_rate_hz * 0.035))
    feedback = min(0.95, max(0.05, decay_sec / 4.0))
    wet = samples[:]
    for index in range(delay_samples, len(wet)):
        wet[index] += wet[index - delay_samples] * feedback
    return clip_samples([
        dry * (1.0 - mix) + reverbed * mix
        for dry, reverbed in zip(samples, wet)
    ])


def apply_gain_db(samples: list[float], gain_db: float) -> list[float]:
    gain = db_to_linear(gain_db)
    return [sample * gain for sample in samples]


def low_pass_filter(samples: list[float], *, sample_rate_hz: int, cutoff_hz: float) -> list[float]:
    cutoff_hz = max(1.0, min(cutoff_hz, (sample_rate_hz / 2.0) - 1.0))
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate_hz
    alpha = dt / (rc + dt)
    output: list[float] = []
    previous = 0.0
    for sample in samples:
        previous = previous + alpha * (sample - previous)
        output.append(previous)
    return output


def high_pass_filter(samples: list[float], *, sample_rate_hz: int, cutoff_hz: float) -> list[float]:
    cutoff_hz = max(1.0, min(cutoff_hz, (sample_rate_hz / 2.0) - 1.0))
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate_hz
    alpha = rc / (rc + dt)
    output: list[float] = []
    previous_output = 0.0
    previous_input = 0.0
    for sample in samples:
        current = alpha * (previous_output + sample - previous_input)
        output.append(current)
        previous_output = current
        previous_input = sample
    return output


def normalize_peak(samples: list[float], *, target_dbfs: float) -> list[float]:
    peak = max((abs(sample) for sample in samples), default=0.0)
    if peak <= 0:
        return samples[:]
    target = db_to_linear(target_dbfs)
    gain = target / peak
    return [sample * gain for sample in samples]


def clip_samples(samples: list[float]) -> list[float]:
    return [max(-1.0, min(1.0, sample)) for sample in samples]


def save_mono_wav(path: str | Path, samples: list[float], *, sample_rate_hz: int) -> Path:
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array("h", [float_to_pcm16(sample) for sample in samples])
    with wave.open(str(wav_path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(pcm.tobytes())
    return wav_path


def float_to_pcm16(sample: float) -> int:
    clipped = max(-1.0, min(1.0, sample))
    if clipped >= 1.0:
        return 32767
    if clipped <= -1.0:
        return -32768
    return int(round(clipped * 32767.0))


def linear_to_dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return round(20.0 * math.log10(value), 8)


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def map_control_to_gain(value: float) -> float:
    value = clamp01(value)
    return 0.5 + value


def get_float(params: dict[str, object], key: str, *, default: float) -> float:
    value = params.get(key, default)
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        return default
    return value
