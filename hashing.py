from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def sha256_state_dict(state_dict_or_module: Any) -> str:
    """Hash a PyTorch state dict deterministically without importing torch eagerly.

    The input may be a module exposing ``state_dict()`` or a mapping of names to
    tensor-like values. Tensors are detached, moved to CPU, made contiguous, and
    hashed together with key, dtype, and shape metadata.
    """

    state: Mapping[str, Any]
    if hasattr(state_dict_or_module, "state_dict"):
        state = state_dict_or_module.state_dict()
    elif isinstance(state_dict_or_module, Mapping):
        state = state_dict_or_module
    else:
        raise TypeError("expected a module with state_dict() or a mapping")

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        shape = tuple(int(x) for x in getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", type(value).__name__))
        digest.update(dtype.encode("utf-8"))
        digest.update(repr(shape).encode("ascii"))
        if hasattr(value, "numpy"):
            digest.update(value.numpy().tobytes(order="C"))
        elif isinstance(value, (bytes, bytearray, memoryview)):
            digest.update(bytes(value))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()
