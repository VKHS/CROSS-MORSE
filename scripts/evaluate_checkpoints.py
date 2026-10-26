#!/usr/bin/env python3
"""Reexecute trusted frozen checkpoints, compare saved predictions, or benchmark."""
from pathlib import Path
import argparse
import json
import math
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_root", type=Path)
    p.add_argument("--trust-local-checkpoints", action="store_true", help="Required: .pt payloads can execute pickle code")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--models", nargs="+", choices=("q0", "qN", "qC", "qD"), default=["q0", "qN", "qC", "qD"])
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=0, help="0: prediction verification only; positive: also repeat timing")
    p.add_argument("--output", required=True, type=Path, help="New report outside the immutable run")
    args = p.parse_args(argv)
    if not args.trust_local_checkpoints:
        p.error("--trust-local-checkpoints is required; never load untrusted .pt files")
    if args.threads < 1 or args.repetitions < 0 or args.warmup < 0 or args.atol < 0 or args.rtol < 0 or not math.isfinite(args.atol) or not math.isfinite(args.rtol):
        p.error("Invalid thread count, repetition count or tolerance")
    try:
        if args.output.exists() or args.output.resolve().is_relative_to(args.run_root.resolve()):
            raise ValueError("Choose a new output outside the immutable run")
        import numpy as np
        import torch
        from cross_morse.reexecution import FrozenRunEvaluator
        torch.set_num_threads(args.threads)
        evaluator = FrozenRunEvaluator(args.run_root, args.device, args.models)
        comparisons = [evaluator.compare(name, args.atol, args.rtol) for name in args.models]
        timings = []
        if args.repetitions:
            for _ in range(args.warmup):
                for name in args.models:
                    evaluator.predict(name)
            rng = np.random.default_rng(evaluator.config["seed"] + 811)
            for repetition in range(args.repetitions):
                for name in rng.permutation(args.models):
                    _, trace = evaluator.predict(str(name))
                    timings.append(dict(trace, predictor=str(name), repetition=repetition))
        summaries = {}
        for name in args.models:
            values = [t["wall_seconds"] for t in timings if t["predictor"] == name]
            if values:
                summaries[name] = {"repetitions": len(values), "mean_seconds": float(np.mean(values)),
                    "median_seconds": float(np.median(values)), "repeat_p95_seconds": float(np.quantile(values, .95)),
                    "observed_maximum_seconds": float(np.max(values))}
        result = {"status": "PASS" if all(x["agrees"] for x in comparisons) else "MISMATCH",
                  "dataset": evaluator.config["dataset"], "source_sha256": evaluator.evidence.identities["source_sha256"],
                  "comparisons": comparisons, "warmup_per_predictor": args.warmup if args.repetitions else 0,
                  "timings": timings, "timing_summary": summaries,
                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads,
                        "device": args.device, "cuda_runtime": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None},
                  "scope": "same prepared graph or corpus; repeat-timing distribution is not an input-population latency tail; no retraining or policy selection"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        print(json.dumps({"status": result["status"], "output": str(args.output), "comparisons": comparisons}, indent=2))
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
