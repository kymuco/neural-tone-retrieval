"""Thin command-line interface over the public API."""

from __future__ import annotations

import argparse
from pathlib import Path

from neural_tone_retrieval.api import ingest_dataset_directory, load_controlled_reamp_config, load_dataset_manifest
from neural_tone_retrieval.examples import write_example_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ntr", description="Neural Tone Retrieval CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Build manifests from raw DI audio")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_ingest = dataset_subparsers.add_parser("ingest", help="Scan WAV files into a manifest")
    dataset_ingest.add_argument("input_dir")
    dataset_ingest.add_argument("output_manifest")
    dataset_ingest.add_argument("--dataset-name", default="neural-tone-retrieval")
    dataset_ingest.add_argument("--dataset-version", default="0.1.0")
    dataset_ingest.add_argument("--pattern", default="*.wav")
    dataset_ingest.add_argument("--non-recursive", action="store_true")
    dataset_ingest.add_argument("--compute-sha256", action="store_true")

    manifest_parser = subparsers.add_parser("manifest", help="Inspect or validate manifests")
    manifest_subparsers = manifest_parser.add_subparsers(dest="manifest_command", required=True)
    manifest_show = manifest_subparsers.add_parser("show", help="Show a manifest summary")
    manifest_show.add_argument("path")
    manifest_validate = manifest_subparsers.add_parser("validate", help="Validate a manifest")
    manifest_validate.add_argument("path")

    config_parser = subparsers.add_parser("config", help="Inspect or validate pipeline configs")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_show = config_subparsers.add_parser("show", help="Show a config summary")
    config_show.add_argument("path")
    config_validate = config_subparsers.add_parser("validate", help="Validate a config")
    config_validate.add_argument("path")

    init_example = subparsers.add_parser("init-example", help="Write an example bundle")
    init_example.add_argument("destination")
    init_example.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset":
        if args.dataset_command == "ingest":
            manifest = ingest_dataset_directory(
                args.input_dir,
                output_manifest_path=args.output_manifest,
                dataset_name=args.dataset_name,
                dataset_version=args.dataset_version,
                pattern=args.pattern,
                recursive=not args.non_recursive,
                compute_sha256=args.compute_sha256,
            )
            print(f"Ingest OK: {args.output_manifest}")
            print(f"dataset={manifest.dataset_name} version={manifest.dataset_version}")
            for key, value in manifest.summary().items():
                print(f"{key}={value}")
            return 0

    if args.command == "manifest":
        manifest = load_dataset_manifest(args.path)
        if args.manifest_command == "validate":
            print(f"Manifest OK: {args.path}")
            print(f"dataset={manifest.dataset_name} version={manifest.dataset_version}")
            return 0
        print_manifest_summary(manifest, Path(args.path))
        return 0

    if args.command == "config":
        config = load_controlled_reamp_config(args.path)
        if args.config_command == "validate":
            print(f"Config OK: {args.path}")
            print(
                "dataset="
                f"{config.dataset.name} version={config.dataset.version} "
                f"top_k={config.retrieval.top_k}"
            )
            return 0
        print_config_summary(config, Path(args.path))
        return 0

    if args.command == "init-example":
        paths = write_example_bundle(args.destination, force=args.force)
        print(f"Wrote example bundle to {Path(args.destination)}")
        print(f"manifest={paths['manifest_path']}")
        print(f"config={paths['config_path']}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


def print_manifest_summary(manifest: object, path: Path) -> None:
    summary = manifest.summary()
    print(f"Manifest: {path}")
    print(f"dataset={manifest.dataset_name}")
    print(f"version={manifest.dataset_version}")
    for key, value in summary.items():
        print(f"{key}={value}")


def print_config_summary(config: object, path: Path) -> None:
    print(f"Config: {path}")
    print(f"dataset={config.dataset.name}")
    print(f"version={config.dataset.version}")
    print(f"manifest_uri={config.dataset.manifest_uri}")
    print(f"raw_di_root={config.dataset.raw_di_root}")
    print(f"output_root={config.dataset.output_root}")
    print(f"selected_splits={','.join(split.value for split in config.dataset.selected_splits)}")
    print(f"sample_rate_hz={config.render.target_sample_rate_hz}")
    print(f"feature_sets={','.join(item.value for item in config.retrieval.feature_sets)}")
    print(f"distance_metric={config.retrieval.distance_metric.value}")
    print(f"top_k={config.retrieval.top_k}")
