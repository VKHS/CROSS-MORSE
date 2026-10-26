#!/usr/bin/env python3
"""Measurement records of the experiments that the eight-stage runner does not produce.

    python -B scripts/records.py index              # hash every deposited set, write MANIFEST.json files and records/index.json
    python -B scripts/records.py check              # verify manifests and trace every summary value to its raw file
    python -B scripts/records.py tables --output-dir paper/generated/official/records
    python -B scripts/records.py list

A set is a directory ``records/<set>/`` named after an entry of ``records/specs.json``:

    summary.csv        one measured value per row: the set's key columns, then quantity, value, unit,
                       source_file (relative to the set) and locator (where in that file the value is)
    raw/               the logs the values are read from (training logs, scheduler accounting, timing traces)
    code/              the scripts and configurations that produced them (optional, recommended)
    provenance.json    experiment, produced_by, hardware, period, protocol
    MANIFEST.json      SHA-256 of every file; written by ``index``, verified by ``check``

Locators:

    json:/path/to/value          RFC 6901 pointer into a JSON file
    csv:column=value;target      first row of a CSV file whose ``column`` equals ``value``; read ``target``
    regex:pattern                first match of a regular expression with one capture group in a text file

``check`` re-reads every value from its raw file and accepts it when the two agree at the precision
printed in summary.csv.  A value that cannot be traced to a deposited raw file fails the check; the
printed tables are built only from sets that pass.  Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"
SPECS = "specs.json"
INDEX = "index.json"
MANIFEST = "MANIFEST.json"
RESERVED = {SPECS, INDEX, "README.md"}


class RecordsError(ValueError):
    """A deposited set is incomplete, altered or not traceable to its raw files."""


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_specs(root: Path) -> dict:
    specs = json.loads((root / SPECS).read_text(encoding="utf-8"))
    if specs.get("schema") != "cross-morse-records-specs-1":
        raise RecordsError("unsupported records/specs.json")
    return specs


def deposited_sets(root: Path, specs: dict) -> list[Path]:
    sets = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if path.name not in specs["sets"]:
            raise RecordsError(f"{path.name}: not a set named in records/specs.json")
        sets.append(path)
    return sets


def set_files(path: Path) -> list[str]:
    return sorted(str(p.relative_to(path)).replace("\\", "/") for p in path.rglob("*")
                  if p.is_file() and p.name != MANIFEST)


def decimals(text: str) -> int:
    text = text.strip().lower()
    mantissa, _, exponent = text.partition("e")
    places = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    return places - (int(exponent) if exponent else 0)


def parse_number(text: str, where: str) -> float:
    try:
        value = float(str(text).strip().replace("−", "-"))
    except ValueError:
        raise RecordsError(f"{where}: not a number: {text!r}") from None
    if not math.isfinite(value):
        raise RecordsError(f"{where}: not finite: {text!r}")
    return value


def json_pointer(document, pointer: str, where: str):
    if pointer in ("", "/"):
        return document
    if not pointer.startswith("/"):
        raise RecordsError(f"{where}: JSON pointer must start with '/': {pointer!r}")
    node = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                raise RecordsError(f"{where}: pointer {pointer!r} does not resolve") from None
        elif isinstance(node, dict) and token in node:
            node = node[token]
        else:
            raise RecordsError(f"{where}: pointer {pointer!r} does not resolve")
    return node


def locate(path: Path, locator: str, where: str) -> str:
    """Return the raw text of the value that ``locator`` points to in ``path``."""
    kind, _, spec = locator.partition(":")
    if kind == "json":
        value = json_pointer(json.loads(path.read_text(encoding="utf-8")), spec, where)
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise RecordsError(f"{where}: pointer resolves to a non-scalar")
        return str(value)
    if kind == "csv":
        match = re.fullmatch(r"([^=;]+)=([^;]*);(.+)", spec)
        if not match:
            raise RecordsError(f"{where}: csv locator must read column=value;target")
        column, key, target = match.groups()
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if column not in row or target not in row:
                    raise RecordsError(f"{where}: columns {column!r}/{target!r} not in {path.name}")
                if row[column] == key:
                    return row[target]
        raise RecordsError(f"{where}: no row with {column} = {key!r} in {path.name}")
    if kind == "regex":
        try:
            pattern = re.compile(spec, re.M)
        except re.error as exc:
            raise RecordsError(f"{where}: invalid regex: {exc}") from None
        if pattern.groups != 1:
            raise RecordsError(f"{where}: regex locator needs exactly one capture group")
        match = pattern.search(path.read_text(encoding="utf-8", errors="strict"))
        if not match:
            raise RecordsError(f"{where}: regex finds nothing in {path.name}")
        return match.group(1)
    raise RecordsError(f"{where}: unknown locator kind {kind!r} (json:, csv:, regex:)")


# ----------------------------------------------------------------------------- one set
def read_summary(path: Path, spec: dict, base_columns: list[str]) -> list[dict]:
    summary = path / "summary.csv"
    if not summary.is_file():
        raise RecordsError(f"{path.name}: summary.csv missing")
    with summary.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = spec["keys"] + base_columns
        if reader.fieldnames != expected:
            raise RecordsError(f"{path.name}: summary.csv columns must be {expected}, found {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise RecordsError(f"{path.name}: summary.csv has no rows")
    return rows


def check_provenance(path: Path, fields: list[str]) -> dict:
    target = path / "provenance.json"
    if not target.is_file():
        raise RecordsError(f"{path.name}: provenance.json missing")
    provenance = json.loads(target.read_text(encoding="utf-8"))
    missing = [f for f in fields if not str(provenance.get(f, "")).strip()]
    if missing:
        raise RecordsError(f"{path.name}: provenance.json lacks {missing}")
    return provenance


def verify_set(path: Path, spec: dict, specs: dict) -> dict:
    """Manifest, provenance and traceability of every value; returns the set's index entry."""
    manifest_path = path / MANIFEST
    if not manifest_path.is_file():
        raise RecordsError(f"{path.name}: MANIFEST.json missing (run scripts/records.py index)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = set_files(path)
    if sorted(manifest.get("files", {})) != files:
        raise RecordsError(f"{path.name}: files differ from MANIFEST.json")
    for name in files:
        if sha256_file(path / name) != manifest["files"][name]:
            raise RecordsError(f"{path.name}: {name} changed after indexing")
    provenance = check_provenance(path, specs["provenance_fields"])
    rows = read_summary(path, spec, specs["summary_columns"])
    seen = set()
    for number, row in enumerate(rows, start=2):
        where = f"{path.name}/summary.csv line {number}"
        if row["quantity"] not in spec["quantities"]:
            raise RecordsError(f"{where}: quantity {row['quantity']!r} is not defined for this set")
        key = tuple(row[k] for k in spec["keys"]) + (row["quantity"],)
        if key in seen:
            raise RecordsError(f"{where}: duplicate {key}")
        seen.add(key)
        for k in spec["keys"]:
            if k == "seed":
                if row["quantity"] in spec.get("per_seed_quantities", []) and not row[k].strip():
                    raise RecordsError(f"{where}: {row['quantity']} is a per-seed quantity; seed is empty")
            elif not row[k].strip():
                raise RecordsError(f"{where}: key {k} is empty")
        value = parse_number(row["value"], where)
        source = Path(row["source_file"])
        if source.is_absolute() or ".." in source.parts or source.as_posix() not in files:
            raise RecordsError(f"{where}: source_file {row['source_file']!r} is not a deposited file of this set")
        if source.parts[0] not in {"raw", "code"}:
            raise RecordsError(f"{where}: source_file must lie under raw/ or code/")
        raw = parse_number(locate(path / source, row["locator"], where), where + " (raw)")
        tolerance = 0.5 * 10 ** (-decimals(row["value"])) + 1e-12 * max(1.0, abs(value))
        if abs(raw - value) > tolerance:
            raise RecordsError(f"{where}: summary value {row['value']} differs from the raw value {raw!r}")
    return {"set": path.name, "verified": True, "rows": len(rows), "files": len(files),
            "manifest_sha256": sha256_file(manifest_path), "manuscript_tables": spec["manuscript_tables"],
            "experiment": provenance.get("experiment"), "produced_by": provenance.get("produced_by")}


def write_manifest(path: Path) -> None:
    files = {name: sha256_file(path / name) for name in set_files(path)}
    (path / MANIFEST).write_text(json.dumps({"schema": "cross-morse-records-manifest-1", "set": path.name,
                                             "files": files}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- tables
def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.6g}"


def build_table(path: Path, spec: dict, specs: dict) -> tuple[list[str], list[list[str]], str]:
    rows = read_summary(path, spec, specs["summary_columns"])
    quantities = list(spec["quantities"])
    mode = spec.get("table", "list")
    keys = [k for k in spec["keys"] if k != "seed"] if mode == "seed_mean_sd" else list(spec["keys"])
    if mode == "configuration_range":
        keys = [k for k in spec["keys"] if k != spec["range_key"]]
    groups: dict[tuple, dict] = {}
    for row in rows:
        group = groups.setdefault(tuple(row[k] for k in keys), {})
        group.setdefault(row["quantity"], []).append(row)
    header, out = list(keys), []
    if mode == "list":
        header += quantities
        for key, values in groups.items():
            out.append(list(key) + [_fmt(values[q][0]["value"]) if q in values else "—" for q in quantities])
        note = "Each cell is a value of summary.csv traced to its raw file by scripts/records.py check."
    elif mode == "seed_mean_sd":
        per_seed = spec.get("per_seed_quantities", [])
        header += [q for q in quantities if q not in per_seed]
        for q in per_seed:
            header += [f"{q} (mean over seeds)", f"{q} SD", "seeds"]
        for key, values in groups.items():
            line = list(key) + [_fmt(next((r["value"] for r in values.get(q, []) if not r["seed"].strip()), None))
                                for q in quantities if q not in per_seed]
            for q in per_seed:
                xs = [float(r["value"]) for r in values.get(q, []) if r["seed"].strip()]
                line += [_fmt(statistics.fmean(xs)) if xs else "—",
                         _fmt(statistics.stdev(xs)) if len(xs) > 1 else "—", str(len(xs))]
            out.append(line)
        note = "Per-method quantities as deposited; the per-seed quantity is summarised by its mean and sample SD over seeds."
    elif mode == "configuration_range":
        central = spec.get("central_configuration", "selected")
        for q in quantities:
            header += [f"{q} ({central})", f"{q} range"]
        for key, values in groups.items():
            line = list(key)
            for q in quantities:
                entries = values.get(q, [])
                chosen = next((r["value"] for r in entries if r[spec["range_key"]] == central), None)
                xs = [float(r["value"]) for r in entries]
                line += [_fmt(chosen), f"[{_fmt(min(xs))}, {_fmt(max(xs))}]" if xs else "—"]
            out.append(line)
        note = (f"The {central!r} configuration gives the central value; the range spans every deposited configuration. "
                "Ranges are configuration ranges, not confidence intervals.")
    elif mode == "ensemble_law":
        header = [k for k in keys if k != "ensemble"] + ["ensemble sizes", "l_inf (nats)", "D_inf (nats)", "max |residual|"]
        by_dataset: dict[tuple, list[tuple[float, float]]] = {}
        for row in rows:
            if row["quantity"] == "loss_nats":
                by_dataset.setdefault(tuple(row[k] for k in keys if k != "ensemble"), []).append(
                    (float(row["ensemble"]), float(row["value"])))
        for key, points in by_dataset.items():
            fit = ensemble_law(points)
            out.append(list(key) + [", ".join(f"{int(e)}" for e, _ in sorted(points)),
                                    _fmt(fit["l_inf"]), _fmt(fit["D_inf"]), _fmt(fit["max_abs_residual"])])
        note = "Least-squares fit of l(E) = l_inf + D_inf / E over the deposited ensemble sizes (Supplementary Table S26b)."
    else:
        raise RecordsError(f"{path.name}: unknown table mode {mode!r}")
    return header, out, note


def ensemble_law(points: list[tuple[float, float]]) -> dict:
    """Ordinary least squares of l = a + b x with x = 1/E."""
    if len(points) < 2:
        raise RecordsError("the ensemble law needs at least two ensemble sizes")
    xs = [1.0 / e for e, _ in points]
    ys = [y for _, y in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise RecordsError("the ensemble law needs distinct ensemble sizes")
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return {"l_inf": a, "D_inf": b, "max_abs_residual": max(abs(y - (a + b * x)) for x, y in zip(xs, ys))}


def render(name: str, spec: dict, header: list[str], rows: list[list[str]], note: str, fmt: str) -> str:
    title = f"{spec['title']} ({', '.join(spec['manuscript_tables'])})"
    if fmt == "md":
        lines = [f"# {title}", "", f"_built from records/{name}/summary.csv; every value traced to its raw file_", "",
                 "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(c.replace("|", r"\|") for c in row) + " |" for row in rows]
        return "\n".join(lines + ["", note, ""])
    if fmt == "csv":
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow([f"# {title}"])
        writer.writerow(header)
        writer.writerows(rows)
        return out.getvalue()
    def tex(s):
        return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
                .replace("#", r"\#"))
    lines = [f"% {title}", f"% built from records/{name}/summary.csv", r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{" + tex(title) + ". " + tex(note) + "}", f"\\label{{tab:records-{name}}}",
             r"\begin{tabular}{" + "l" * len(header) + "}", r"\hline", " & ".join(tex(h) for h in header) + r" \\",
             r"\hline"] + [" & ".join(tex(c) for c in row) + r" \\" for row in rows] + [r"\hline", r"\end{tabular}",
                                                                                     r"\end{table}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- commands
def cmd_index(root: Path) -> int:
    specs = load_specs(root)
    entries = []
    for path in deposited_sets(root, specs):
        write_manifest(path)
        entries.append(verify_set(path, specs["sets"][path.name], specs))
    write_index(root, specs, entries)
    print(f"indexed {len(entries)} record set(s) of {len(specs['sets'])}")
    return 0


def write_index(root: Path, specs: dict, entries: list[dict]) -> None:
    index = {"schema": "cross-morse-records-index-1", "sets_defined": list(specs["sets"]),
             "sets": sorted(entries, key=lambda e: list(specs["sets"]).index(e["set"]))}
    (root / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def cmd_check(root: Path) -> int:
    specs = load_specs(root)
    problems, entries = [], []
    for path in deposited_sets(root, specs):
        try:
            entries.append(verify_set(path, specs["sets"][path.name], specs))
        except (RecordsError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            problems.append(str(exc))
    index_path = root / INDEX
    recorded = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"sets": []}
    if [e["set"] for e in recorded.get("sets", [])] != [e["set"] for e in entries] or any(
            a.get("manifest_sha256") != b["manifest_sha256"] for a, b in zip(recorded.get("sets", []), entries)):
        problems.append("records/index.json does not match the deposited sets (run scripts/records.py index)")
    for problem in problems:
        print("CHECK FAILED:", problem, file=sys.stderr)
    if problems:
        return 1
    print(f"{len(entries)} of {len(specs['sets'])} measurement record sets deposited; every value traced to its raw file")
    return 0


def cmd_tables(root: Path, output_dir: Path) -> int:
    specs = load_specs(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for path in deposited_sets(root, specs):
        spec = specs["sets"][path.name]
        verify_set(path, spec, specs)
        header, rows, note = build_table(path, spec, specs)
        for fmt in ("tex", "md", "csv"):
            (output_dir / f"{path.name}.{fmt}").write_text(render(path.name, spec, header, rows, note, fmt), encoding="utf-8")
        built.append(path.name)
    (output_dir / "INDEX.json").write_text(json.dumps({"record_tables": built}, indent=1) + "\n", encoding="utf-8")
    print(f"built {len(built)} record table(s) under {output_dir}")
    return 0


def cmd_list(root: Path) -> int:
    specs = load_specs(root)
    present = {p.name for p in deposited_sets(root, specs)}
    for name, spec in specs["sets"].items():
        state = "deposited" if name in present else "pending"
        print(f"{name:18s} {state:9s} {', '.join(spec['manuscript_tables'])}")
    return 0


def summary_state(root: Path = RECORDS) -> dict:
    """For the availability statement: which sets are defined and which are indexed."""
    specs = load_specs(root)
    index_path = root / INDEX
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"sets": []}
    verified = [e["set"] for e in index.get("sets", []) if e.get("verified")]
    return {"defined": list(specs["sets"]), "verified": verified,
            "tables": {name: spec["manuscript_tables"] for name, spec in specs["sets"].items()}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("index", "check", "tables", "list"))
    parser.add_argument("--root", type=Path, default=RECORDS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper" / "generated" / "official" / "records")
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            return cmd_index(args.root)
        if args.command == "check":
            return cmd_check(args.root)
        if args.command == "tables":
            return cmd_tables(args.root, args.output_dir)
        return cmd_list(args.root)
    except (RecordsError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
