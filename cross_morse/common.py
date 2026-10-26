"""Small, explicit run-state and reproducibility helpers."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import signal
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"[checkpoint] signal {signum}; stopping at the next safe boundary", flush=True)


def install_signal_handlers():
    for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)


def stop_requested():
    return STOP_REQUESTED


def stop_at_boundary():
    if STOP_REQUESTED:
        raise InterruptedError("Checkpoint saved at safe boundary; resume the same output directory.")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as f:
        json.dump(jsonable(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def precision_context(config, device):
    precision = config["model"]["precision"]
    if precision == "bf16":
        if torch.device(device).type != "cuda":
            raise ValueError("bf16 plant is registered for CUDA; use the explicit fp32 smoke config on CPU.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support the registered bf16 plant precision.")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision != "fp32":
        raise ValueError(f"Unsupported plant precision: {precision}")
    return contextlib.nullcontext()


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def seconds():
    return time.perf_counter()
