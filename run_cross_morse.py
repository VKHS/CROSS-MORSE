#!/usr/bin/env python3
"""Run the CROSS-MORSE lifecycle for one registered dataset and split."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import tempfile
import traceback

from cross_morse import __version__


DATASET_CONFIG_STEMS = {
    "ogbn-arxiv": "arxiv",
    "ogbn-products": "products",
    "ogbn-proteins": "proteins",
    "Roman-empire": "roman_empire",
    "PascalVOC-SP": "pascalvoc_sp",
    "COCO-SP": "coco_sp",
    "Peptides-func": "peptides_func",
}
PROFILES = ("smoke", "pilot", "full")
STAGES = ("all", "prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate")
ROOT = Path(__file__).resolve().parent
SNAPSHOT_ENV_KEYS = (
    "CM_BASE", "CM_DATA_ROOT", "CM_VENV_DIR", "CM_ACCOUNT", "CM_STORAGE_PROJECT",
    "CM_PARTITION", "CM_BUILD_MODULE", "CM_PYTHON_MODULE", "CM_EXPECT_ARCH",
    "CM_THREADS", "CM_TORCH_SPEC", "CM_TORCH_INDEX", "CM_DATASET", "CM_SPLIT_INDEX",
)


def _reject_nonfinite(value):
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def resolve_configuration(*, config_path=None, dataset=None, profile=None,
                          split_index=None, default_dataset=None):
    """Read and validate routing using only the standard library.

    Split overrides become part of the effective configuration and therefore
    its run identity. Synthetic fixtures carry an explicit benchmark_dataset
    label for routing, while dataset remains synthetic in all saved records.
    """
    if default_dataset is not None and default_dataset not in DATASET_CONFIG_STEMS:
        raise ValueError(f"Unknown wrapper dataset: {default_dataset}")
    if dataset is not None and dataset not in DATASET_CONFIG_STEMS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if dataset is not None and default_dataset is not None and dataset != default_dataset:
        raise ValueError(f"This entry point is for {default_dataset}, not {dataset}")
    selected = dataset or default_dataset
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if config_path is None:
        if selected is None:
            raise ValueError("Provide --dataset or an explicit --config")
        config_path = ROOT / "configs" / f"{DATASET_CONFIG_STEMS[selected]}_{profile or 'pilot'}.json"
    else:
        config_path = Path(config_path).expanduser()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    config = copy.deepcopy(config)
    actual_dataset = config.get("dataset")
    if actual_dataset == "synthetic":
        benchmark = config.get("benchmark_dataset")
        # Preserve the original arxiv wrapper's explicit synthetic-config use.
        if benchmark is None and default_dataset == "ogbn-arxiv":
            benchmark = "ogbn-arxiv"
            config["benchmark_dataset"] = benchmark
        if benchmark not in DATASET_CONFIG_STEMS:
            raise ValueError("Synthetic configs require a canonical benchmark_dataset routing label")
        if config.get("scientific_run") is not False:
            raise ValueError("Synthetic fixtures must explicitly set scientific_run=false")
        if config.get("profile") != "smoke":
            raise ValueError("Synthetic fixtures must use profile='smoke'")
    else:
        benchmark = actual_dataset
        if benchmark not in DATASET_CONFIG_STEMS:
            raise ValueError(f"Unsupported dataset in config: {actual_dataset!r}")
        if config.get("benchmark_dataset", benchmark) != benchmark:
            raise ValueError("benchmark_dataset conflicts with the real dataset in config")
        if config.get("profile") == "smoke":
            raise ValueError("Smoke profiles must explicitly identify synthetic data")
    if selected is not None and benchmark != selected:
        raise ValueError(f"Dataset/config mismatch: selected {selected}, config describes {benchmark}")
    if profile is not None and config.get("profile") != profile:
        raise ValueError(f"Profile/config mismatch: selected {profile}, config uses {config.get('profile')!r}")
    if config.get("profile") not in PROFILES:
        raise ValueError("Config profile must be smoke, pilot or full")
    data = config.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("The data configuration must be an object")
    if benchmark == "Roman-empire":
        index = split_index if split_index is not None else data.get("split_index", 0)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 9:
            raise ValueError("Roman-empire split_index must be an integer from 0 through 9")
        data["split_index"] = index
    elif split_index is not None or "split_index" in data:
        raise ValueError("--split-index/data.split_index is supported only for Roman-empire")
    return config


def _atomic_json(path, value):
    """Write launcher metadata without importing the numerical runtime."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_launcher_snapshot(source, suite, config_path, profile, data_root,
                             runtime_shell, *, dataset=None, split_index=None, capture_split=True):
    """Capture a new source/config/profile snapshot for a suite or campaign.

    Inputs may contain spaces. Existing snapshots are never overwritten, and a
    snapshot cannot be written into its source tree or contain the source tree.
    The caller creates a unique suite and holds its launcher lock.
    """
    source = Path(source).expanduser().resolve()
    suite = Path(suite).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    runtime_shell = Path(runtime_shell).resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory is absent: {source}")
    if suite.is_relative_to(source) or source.is_relative_to(suite):
        raise ValueError("Output suite/campaign must be outside and must not contain the source tree")
    if (suite / "source").exists() or (suite / "source").is_symlink():
        raise FileExistsError(f"Source snapshot already exists: {suite / 'source'}")
    config = resolve_configuration(config_path=config_path, dataset=dataset,
                                   profile=profile, split_index=split_index)
    benchmark = config.get("benchmark_dataset", config["dataset"])
    split = config["data"].get("split_index")
    suite.mkdir(parents=True, exist_ok=True)
    (suite / "launcher").mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".source_build_", dir=suite))
    skip = {"__pycache__", ".git", ".venv", "venv", "datasets", "data", "outputs",
            "runs", "cache", "upload", "analysis", "logs", "dist", "build", "exports"}
    extensions = {".py", ".sh", ".json", ".toml", ".txt", ".md", ".yaml", ".yml", ".example", ".cff"}
    try:
        for folder, directories, names in os.walk(source, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in skip
                                     and not d.startswith((".", "venv"))
                                     and not (Path(folder) / d).is_symlink())
            for name in sorted(names):
                path = Path(folder) / name
                if (path.is_symlink() or not path.is_file()
                    or (path.suffix not in extensions and path.name not in {"LICENSE", "NOTICE", "Makefile"})
                    or path.stat().st_size > 10 * 1024**2):
                    continue
                target = temporary / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        (temporary / "configs").mkdir(exist_ok=True)
        shutil.copy2(config_path, temporary / "configs/submitted.json")
        _atomic_json(temporary / "configs/resolved.json", config)
        effective = {key: os.environ.get(key, "") for key in SNAPSHOT_ENV_KEYS}
        effective["CM_DATASET"] = benchmark
        effective["CM_SPLIT_INDEX"] = str(split) if split is not None and capture_split else ""
        effective["CM_DATA_ROOT"] = str(Path(data_root).expanduser().resolve())
        (temporary / "configs/active.env.example").write_text(
            "# Effective cluster environment captured for this run; exported overrides still win.\n"
            + "".join(f"{key}={shlex.quote(value)}\n" for key, value in effective.items()), encoding="utf-8")
        # This may be Slurm's spooled master shell; capture those executed bytes.
        shutil.copy2(runtime_shell, temporary / "run_cross_morse.sh")
        hashes = {p.relative_to(temporary).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(temporary.rglob("*")) if p.is_file()}
        manifest = {
            "launch_schema_version": 2, "dataset": benchmark,
            "data_kind": config["dataset"], "profile": profile, "split_index": split,
            "data_root": effective["CM_DATA_ROOT"], "original_source": str(source),
            "original_config": str(config_path),
            "original_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "account": os.environ.get("SLURM_JOB_ACCOUNT", effective["CM_ACCOUNT"]),
            "storage_project": effective["CM_STORAGE_PROJECT"], "base": effective["CM_BASE"],
        }
        os.replace(temporary, suite / "source")
        _atomic_json(suite / "launcher/source_sha256.json", hashes)
        _atomic_json(suite / "launcher/run_manifest.json", manifest)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_launcher_snapshot(suite):
    """Reject missing, changed, extra or unsafe files in a captured source."""
    suite = Path(suite).expanduser().resolve()
    source = suite / "source"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Suite has no regular source snapshot")
    hashes = json.loads((suite / "launcher/source_sha256.json").read_text())
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Source snapshot hash inventory is empty or invalid")
    bad = []
    for name, expected in hashes.items():
        if not isinstance(name, str):
            raise ValueError("Source snapshot contains an unsafe hash entry")
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("Source snapshot contains an unsafe hash entry")
        path = source / relative
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad.append(name)
    current = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            bad.append(path.relative_to(source).as_posix())
        elif path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}:
            current.add(path.relative_to(source).as_posix())
    bad.extend(sorted(current - set(hashes)))
    if bad:
        raise ValueError("Source snapshot changed; refusing mixed-code resume: " + ", ".join(sorted(set(bad))))
    manifest = json.loads((suite / "launcher/run_manifest.json").read_text())
    config = resolve_configuration(config_path=source / "configs/resolved.json",
                                   dataset=manifest.get("dataset"), profile=manifest.get("profile"))
    if config["data"].get("split_index") != manifest.get("split_index"):
        raise ValueError("Suite split metadata disagrees with its effective config")
    return manifest


def create_roman_campaign(source, campaign, config_path, profile, data_root, runtime_shell):
    """Freeze one common source/config before the ten Slurm array tasks start."""
    source = Path(source).expanduser().resolve()
    campaign = Path(campaign).expanduser().resolve()
    if campaign.is_relative_to(source) or source.is_relative_to(campaign):
        raise ValueError("Campaign directory must be outside and must not contain the source tree")
    if campaign.exists() or campaign.is_symlink():
        raise FileExistsError("Campaign directory already exists; use 'submit campaign resume' to continue it")
    campaign.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + campaign.name + ".build_", dir=campaign.parent))
    try:
        create_launcher_snapshot(source, temporary, config_path, profile, data_root, runtime_shell,
                                 dataset="Roman-empire", split_index=0, capture_split=False)
        _atomic_json(temporary / "campaign.json", {
            "schema_version": 1, "dataset": "Roman-empire", "profile": profile,
            "split_indices": list(range(10)), "run_directory_pattern": "split_{split_index:02d}",
            "summary_command": "python scripts/summarize_roman_splits.py CAMPAIGN --output NEW_REPORT.json",
            "statistical_scope": "Ten official split results; procedure-level variation is descriptive.",
        })
        verify_launcher_snapshot(temporary)
        if campaign.exists() or campaign.is_symlink():
            raise FileExistsError("Campaign destination appeared during preparation; refusing replacement")
        os.rename(temporary, campaign)
        return campaign
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_roman_campaign(campaign):
    campaign = Path(campaign).expanduser().resolve()
    manifest = verify_launcher_snapshot(campaign)
    record = json.loads((campaign / "campaign.json").read_text())
    if (manifest.get("dataset") != "Roman-empire" or record.get("dataset") != "Roman-empire"
        or record.get("split_indices") != list(range(10)) or record.get("profile") != manifest.get("profile")
        or record.get("run_directory_pattern") != "split_{split_index:02d}"):
        raise ValueError("Campaign must identify the ten Roman-empire splits and one common profile")
    return record


def main(argv=None, *, default_dataset=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--gpu-health-check", action="store_true",
                        help="check CUDA sparse/dense operations without loading a dataset")
    parser.add_argument("--dataset", choices=tuple(DATASET_CONFIG_STEMS),
                        help="canonical benchmark name; must match the config and dataset wrapper")
    parser.add_argument("--profile", choices=PROFILES,
                        help="select a supplied config (default pilot) or check an explicit config's profile")
    parser.add_argument("--config", type=Path, help="explicit JSON config; overrides profile-based path selection")
    parser.add_argument("--split-index", type=int, choices=range(10),
                        help="Roman-empire split 0..9; recorded in the effective configuration")
    parser.add_argument("--print-config", action="store_true",
                        help="print validated effective JSON and exit without runtime imports or writes")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if default_dataset is not None and args.dataset is not None and args.dataset != default_dataset:
        parser.error(f"This entry point is for {default_dataset}, not {args.dataset}")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.gpu_health_check and args.print_config:
        parser.error("--gpu-health-check and --print-config are separate standalone operations")
    config = None
    if not args.gpu_health_check:
        try:
            config = resolve_configuration(config_path=args.config, dataset=args.dataset,
                                           profile=args.profile, split_index=args.split_index,
                                           default_dataset=default_dataset)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        if args.print_config:
            print(json.dumps(config, indent=2, allow_nan=False))
            return 0
        if args.output_dir is None:
            parser.error("--output-dir is required for a run")

    import torch
    torch.set_num_threads(args.threads)
    if args.gpu_health_check:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; the launcher will not silently use the CPU")
        device = torch.device("cuda")
        x = torch.randn(256, 256, device=device)
        indices = torch.arange(256, device=device).repeat(2, 1)
        sparse = torch.sparse_coo_tensor(indices, torch.ones(256, device=device), (256, 256)).coalesce()
        result = torch.sparse.mm(sparse, x)
        dense = x @ x.T
        torch.cuda.synchronize()
        if not torch.allclose(result, x) or not torch.isfinite(dense).all():
            raise RuntimeError("CUDA sparse/dense health check failed")
        print(json.dumps({"cuda": torch.version.cuda, "torch": torch.__version__,
                          "gpu": torch.cuda.get_device_name(), "bf16": torch.cuda.is_bf16_supported(),
                          "status": "PASS"}))
        return 0
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use an explicit FP32 smoke profile with --device cpu.")
    from cross_morse.common import install_signal_handlers, atomic_json, seed_everything
    from cross_morse.pipeline import Pipeline
    from cross_morse.learning import TrainingInterrupted
    install_signal_handlers()
    seed_everything(config["seed"])
    try:
        pipeline = Pipeline(config, args.data_root, args.output_dir, args.device, args.resume)
        pipeline.run(args.stage)
        return 0
    except (InterruptedError, TrainingInterrupted) as exc:
        atomic_json(args.output_dir / "interruption.json", {
            "status": "CHECKPOINTED_INTERRUPTION", "reason": str(exc),
            "resume": "Use the same config/source/dataset/split/output directory with --resume"})
        print(f"[checkpoint] {exc}", file=sys.stderr)
        return 75
    except Exception as exc:
        atomic_json(args.output_dir / "failure_summary.json", {
            "status": "FAILED", "type": type(exc).__name__, "reason": str(exc),
            "traceback": traceback.format_exc(), "scientific_result": "NOT_FABRICATED"})
        raise


if __name__ == "__main__":
    sys.exit(main())
