from __future__ import annotations

from pathlib import Path

from neural_tone_retrieval.cli import main
from neural_tone_retrieval.examples import (
    build_example_controlled_reamp_config,
    build_example_dataset_manifest,
    write_example_bundle,
)
from neural_tone_retrieval.serde import (
    load_controlled_reamp_config,
    load_dataset_manifest,
    render_controlled_reamp_config_toml,
    save_dataset_manifest,
)


def test_manifest_round_trip_json(tmp_path: Path) -> None:
    manifest = build_example_dataset_manifest()
    manifest_path = tmp_path / "example_dataset_manifest.json"

    save_dataset_manifest(manifest, manifest_path)
    loaded = load_dataset_manifest(manifest_path)

    assert loaded.dataset_name == manifest.dataset_name
    assert loaded.summary() == manifest.summary()


def test_controlled_reamp_config_loads_from_toml(tmp_path: Path) -> None:
    config = build_example_controlled_reamp_config()
    config_path = tmp_path / "controlled_reamp.toml"
    config_path.write_text(render_controlled_reamp_config_toml(config), encoding="utf-8")

    loaded = load_controlled_reamp_config(config_path)

    assert loaded.dataset.name == config.dataset.name
    assert loaded.retrieval.top_k == 5
    assert loaded.render.target_sample_rate_hz == 44_100


def test_write_example_bundle_creates_expected_files(tmp_path: Path) -> None:
    paths = write_example_bundle(tmp_path)

    assert paths["manifest_path"].exists()
    assert paths["config_path"].exists()


def test_cli_init_example_and_validate(tmp_path: Path, capsys) -> None:
    destination = tmp_path / "bundle"

    assert main(["init-example", str(destination)]) == 0
    init_output = capsys.readouterr().out
    assert "Wrote example bundle" in init_output

    manifest_path = destination / "example_dataset_manifest.json"
    config_path = destination / "controlled_reamp.toml"

    assert main(["manifest", "validate", str(manifest_path)]) == 0
    manifest_output = capsys.readouterr().out
    assert "Manifest OK" in manifest_output

    assert main(["config", "show", str(config_path)]) == 0
    config_output = capsys.readouterr().out
    assert "feature_sets=mel,mfcc,spectral" in config_output
