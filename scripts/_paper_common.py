"""Shared helpers for the manuscript registry checks.

Everything here is standard-library Python. The helpers re-implement the three
registry formulas of ``cross_morse/core.py`` (executed hops, identity mass and
the spectral response of a registered action) so that the manuscript constants
can be verified in an environment without PyTorch, and so that the comparison
between the two implementations is itself a test (``tests/test_paper_registry.py``).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
REGISTERED = PAPER / "registered_constants.json"
REPORTED = PAPER / "reported_results.json"

DATASETS = ("ogbn-arxiv", "ogbn-proteins", "ogbn-products", "Roman-empire", "PascalVOC-SP", "COCO-SP", "Peptides-func")
STEMS = {"ogbn-arxiv": "arxiv", "ogbn-proteins": "proteins", "ogbn-products": "products", "Roman-empire": "roman_empire",
         "PascalVOC-SP": "pascalvoc_sp", "COCO-SP": "coco_sp", "Peptides-func": "peptides_func"}
BCE_DATASETS = ("ogbn-proteins", "Peptides-func")
MULTI_GRAPH = ("PascalVOC-SP", "COCO-SP", "Peptides-func")

# Manuscript route names <-> implementation route identifiers (cross_morse/policies.py: make_registry).
ROUTE_NAMES = {"nominal": "nominal", "best_depth": "best depth", "learned_stop": "learned stop", "phase": "phase",
               "fixed_sequence": "fixed sequence", "mpc_ol": "MPC-OL", "one_step": "one-step", "radius": "radius", "mpc_rh": "MPC-RH"}


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registered() -> dict:
    return read_json(REGISTERED)


def load_reported() -> dict:
    return read_json(REPORTED)


def full_config(dataset: str) -> dict:
    return read_json(ROOT / "configs" / f"{STEMS[dataset]}_full.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_source_fingerprint(root: Path = ROOT) -> str:
    """Identical to ``cross_morse.pipeline.source_fingerprint`` without importing it.

    The runtime identity covers ``run_cross_morse*.py`` and ``cross_morse/*.py``:
    the canonical JSON of {relative path: sha256} is hashed with sha256.
    """
    files = sorted(root.glob("run_cross_morse*.py")) + sorted((root / "cross_morse").glob("*.py"))
    table = {str(p.relative_to(root)): sha256_file(p) for p in files if p.exists()}
    payload = json.dumps(table, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


# ----------------------------------------------------------------------------------------------
# The executable registry as written in cross_morse/core.py, read without importing torch
# ----------------------------------------------------------------------------------------------
def parse_actions_from_core(core_path: Path = ROOT / "cross_morse" / "core.py") -> list[dict]:
    """Return the ``ACTIONS`` tuple of core.py as plain dictionaries.

    The parser evaluates only literal ``Action(...)`` constructor calls, so it
    cannot execute code; a changed registry shape raises instead of guessing.
    """
    tree = ast.parse(core_path.read_text(encoding="utf-8"), filename=str(core_path))
    fields = ["id", "name", "rho", "nu", "alpha", "eta", "order", "probability", "mode"]
    defaults = {"eta": None, "order": 8, "probability": 0.0, "mode": "transport"}
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "ACTIONS" for t in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Tuple):
            raise ValueError("ACTIONS must be a literal tuple")
        actions = []
        for call in value.elts:
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "Action"):
                raise ValueError("ACTIONS entries must be literal Action(...) calls")
            record = dict(defaults)
            for name, arg in zip(fields, call.args):
                record[name] = ast.literal_eval(arg)
            for keyword in call.keywords:
                record[keyword.arg] = ast.literal_eval(keyword.value)
            missing = [name for name in fields[:5] if name not in record]
            if missing:
                raise ValueError(f"Action without {missing}")
            actions.append(record)
        return actions
    raise ValueError("ACTIONS registry not found in core.py")


# ----------------------------------------------------------------------------------------------
# Registry formulas (main Eqs. (9)-(10); Supplementary Table S6b)
# ----------------------------------------------------------------------------------------------
def resolvent_hops(eta: float, order: int = 8) -> float:
    """Expected hop count of the mass-corrected truncated resolvent kernel."""
    return sum(t * eta * (1 - eta) ** t for t in range(order + 1))


def kernel_hops(rho: float, eta: float | None, order: int = 8) -> float:
    return 1.0 if eta is None else (1 - rho) + rho * resolvent_hops(eta, order)


def executed_hops(rho: float, nu: float, eta: float | None, order: int = 8) -> float:
    """Main Eq. (10): h_bar_a = nu [(1-rho) + rho sum_t t eta (1-eta)^t]."""
    return nu * kernel_hops(rho, eta, order)


def identity_mass(eta: float | None, order: int = 8) -> float | None:
    """(1-eta)^(T+1): the diagonal mass retained by the truncated resolvent."""
    return None if eta is None else (1 - eta) ** (order + 1)


def executed_identity_coefficient(rho: float, nu: float, eta: float | None, order: int = 8) -> float:
    """Coefficient of P^0 in the executed operator T_a = (1-nu) I + nu K_a."""
    if eta is None:
        return 1 - nu
    return (1 - nu) + nu * rho * (eta + identity_mass(eta, order))


def spectral_response(rho: float, nu: float, eta: float | None, lam: float, order: int = 8) -> float:
    """g_a(lambda) for lambda an eigenvalue of L_sym (so P has eigenvalue 1 - lambda)."""
    p = 1.0 - lam
    kernel = p
    if eta is not None:
        resolvent = eta * sum((1 - eta) ** t * p ** t for t in range(order + 1)) + (1 - eta) ** (order + 1)
        kernel = (1 - rho) * p + rho * resolvent
    return (1 - nu) + nu * kernel


def band_sensitivity(action: dict, nominal: dict, bands: int = 6, grid: int = 1025, order: int = 8) -> list[float]:
    """Supplementary Figure S15b: s_ab = max_{lambda in I_b} |g_a(lambda) - g_A0(lambda)| on a uniform partition of [0, 2]."""
    values = []
    for b in range(bands):
        lo, hi = 2.0 * b / bands, 2.0 * (b + 1) / bands
        best = 0.0
        for i in range(grid):
            lam = lo + (hi - lo) * i / (grid - 1)
            diff = abs(spectral_response(action["rho"], action["nu"], action["eta"], lam, order)
                       - spectral_response(nominal["rho"], nominal["nu"], nominal["eta"], lam, order))
            best = max(best, diff)
        values.append(best)
    return values


# ----------------------------------------------------------------------------------------------
# Statistical formulas used by the admission and economics tables
# ----------------------------------------------------------------------------------------------
def bernstein_penalty(S2: float, J: int, B: float, delta: float) -> float:
    """Second and third terms of main Eq. (27)."""
    log_term = math.log(2.0 / delta)
    return math.sqrt(2.0 * S2 * log_term / J) + 14.0 * B * log_term / (3.0 * (J - 1))


def minimum_admission_J(D_bar: float, epsilon_a: float, B: float = 0.10, delta: float = 0.025) -> float:
    """Main Corollary 2: J > 1 + 14 B log(2/delta) / (3 (|D_bar| - epsilon_a))."""
    return 1.0 + 14.0 * B * math.log(2.0 / delta) / (3.0 * (abs(D_bar) - epsilon_a))


def exchange_rate_mnats(abs_delta: float, expected_cost: float, comparator_cost: float) -> float:
    """Main Definition 2: lambda* = |Delta| / (E C(q_D) - E C(q_C)), in millinats."""
    return 1000.0 * abs_delta / (expected_cost - comparator_cost)


def amortization_thousand_input_seconds(qualification_gpu_h: float, expected_cost: float, comparator_cost: float, theta: float = 0.1) -> float:
    """Main Definition 2: M* t_nom = 3600 C_qual / [theta (E C(q_D) - E C(q_C))], in 10^3 input-seconds."""
    return 3600.0 * qualification_gpu_h / (theta * (expected_cost - comparator_cost)) / 1000.0


def printed_half_unit(value) -> float:
    """Half of the last printed decimal of a transcribed value, as a tolerance for rounding."""
    if isinstance(value, bool):
        raise TypeError("a boolean has no printed precision")
    if isinstance(value, int):
        return 0.5
    if float(value) == 0.0:
        return 0.0  # an exact zero (the comparator-return case) is byte-identical by construction
    text = repr(float(value))
    if "e" in text or "E" in text:
        mantissa, exponent = text.lower().split("e")
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return 0.5 * 10 ** (int(exponent) - decimals)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return 0.5 * 10 ** (-decimals)
