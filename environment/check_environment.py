#!/usr/bin/env python3
"""Check the observed CPU package pins without importing numerical libraries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_spec(directory: Path) -> tuple[dict, dict[str, str]]:
    metadata = json.loads((directory / "observed_environment.json").read_text(encoding="utf-8"))
    lock = directory / "requirements-cpu.lock"
    if hashlib.sha256(lock.read_bytes()).hexdigest() != metadata["requirements_sha256"]:
        raise ValueError("CPU lock bytes differ from the observed environment record")
    pins = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match or normalize(match[1]) in pins:
            raise ValueError(f"Expected a unique exact package pin: {line}")
        pins[normalize(match[1])] = match[2]
    observed = {normalize(name): version for name, version in metadata["packages"].items()}
    if pins != observed or len(pins) != metadata["package_count"]:
        raise ValueError("CPU package pins differ from the observed distribution inventory")
    if not pins.get("torch", "").endswith("+cpu"):
        raise ValueError("This installer requires the observed CPU-only PyTorch distribution")
    return metadata, pins


def platform_differences(metadata: dict) -> list[str]:
    differences = []
    for key, actual in (("system", platform.system()), ("machine", platform.machine())):
        expected = metadata["platform"][key]
        if actual != expected:
            differences.append(f"{key}: expected {expected}, found {actual}")
    expected = metadata["python"]["version_info"][:2]
    if list(sys.version_info[:2]) != expected:
        differences.append(f"Python major/minor: expected {'.'.join(map(str, expected))}, found {platform.python_version()}")
    return differences


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap", action="store_true", help="check Python/platform and print the CPU Torch pin for the installer")
    args = parser.parse_args(argv)
    try:
        metadata, pins = load_spec(args.spec_dir)
        differences = platform_differences(metadata)
        if args.bootstrap:
            if differences:
                raise ValueError("; ".join(differences))
            print("torch==" + pins["torch"])
            return 0
        installed = {normalize(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}
        for name, expected in sorted(pins.items()):
            actual = installed.get(name)
            if actual != expected:
                differences.append(f"{name}: expected {expected}, found {actual if actual is not None else 'missing'}")
        print(json.dumps({"status": "PINNED_PACKAGES_MATCH" if not differences else "ENVIRONMENT_DIFFERS",
                          "package_count": len(pins), "differences": differences,
                          "python": platform.python_version(), "observed_python": metadata["python"]["version"],
                          "python_patch_matches": platform.python_version() == metadata["python"]["version"],
                          "scope": "package versions and Python major/minor/platform; not wheel hashes, native-library identity or GPU compatibility"}, indent=2))
        return 1 if differences else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
