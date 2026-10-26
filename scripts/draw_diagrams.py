#!/usr/bin/env python3
"""Draw the repository's explanatory block diagrams and workflow figures.

The figures are self-contained SVG with optional PNG copies. Registered numeric
settings are read from ``paper/registered_constants.json``; execution summaries
refer to the actual methods named in ``docs/assets/README.md``. These explanatory
figures contain no benchmark measurements. System fonts are used without remote
font requests. SVG bytes are deterministic; raster appearance depends on the
installed renderer and fonts.

Figures
    architecture        the frozen plant, label-free control loop and
                        offline-fitted models fixed during execution
    lifecycle           the eight sealed stages with their label roles,
                        candidate/comparator branches and final policy lock
    control_depth       transport operators, native versus inserted sites
                        and the ten-action registry
    policy_families     the nine policy families, their information contracts
                        and the comparator -> deployed-rule decision chain
    review_workflow     the evidence tiers a reviewer can execute, with the
                        commands and the artifacts each tier checks
    evidence_chain      from the source identity through Arrhenius execution, the
                        run-archive contract and the paper build to the manuscript
    decision_chain      the seven decision statistics of a run, the sealed file each
                        is read from and the table cell it fills

Usage
    python -B scripts/draw_diagrams.py --output-dir docs/assets
    python -B scripts/draw_diagrams.py --output-dir docs/assets --png
    python -B scripts/draw_diagrams.py --check          # regenerate in memory and compare

PNG rasterization uses an installed CairoSVG, Inkscape or Playwright Chromium;
it never downloads a renderer. Explicit ``--png`` fails if rasterization fails.
``--check --png`` additionally checks each PNG's embedded SVG-source hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

FONT = "'DejaVu Sans', Arial, 'Helvetica Neue', sans-serif"
MONO = "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, 'DejaVu Sans Mono', monospace"

# ---------------------------------------------------------------------------
# Palette: one hue per semantic role, used identically in every figure.
INK = "#172b3a"
MUTED = "#526171"
LINE = "#7e8e9b"
BORDER = "#d8e1e8"
PANEL_LABEL = "#344b60"

PLANT = ("#315e91", "#edf3fa", "#b8cce3")  # frozen predictor
SENSE = ("#16776b", "#edf7f4", "#b9dcd2")  # label-free estimation
MODEL = SENSE                              # fitted offline, fixed at runtime
POLICY = ("#94611c", "#fcf5e8", "#e7d0a7") # decisions and actions
GATE = POLICY
ART = ("#526171", "#f4f6f8", "#d1dbe3")    # data and evidence
PASS = SENSE

LEGEND = [
    ("Frozen predictor", PLANT),
    ("Label-free runtime", SENSE),
    ("Learned, then frozen", MODEL),
    ("Actions and policies", POLICY),
    ("Labels, gates, decisions", GATE),
    ("Sealed artifacts and tools", ART),
]


# ---------------------------------------------------------------------------
# Text measurement from an embedded advance-width table (no font library needed).
# Advance widths per em (the larger of Inter and DejaVu Sans, the widest common fallback), so that
# the layout is deterministic on every machine and errs on the side of extra room.
_ADVANCE_NORMAL = {
    ' ': 0.318, '!': 0.401, '"': 0.466, '#': 0.838, '$': 0.642, '%': 0.982, '&': 0.78, "'": 0.3, '(': 0.39,
    ')': 0.39, '*': 0.501, '+': 0.838, ',': 0.318, '-': 0.46, '.': 0.318, '/': 0.36, '0': 0.636, '1': 0.636,
    '2': 0.636, '3': 0.636, '4': 0.646, '5': 0.636, '6': 0.636, '7': 0.636, '8': 0.636, '9': 0.636, ':': 0.337,
    ';': 0.337, '<': 0.838, '=': 0.838, '>': 0.838, '?': 0.531, '@': 1.0, 'A': 0.69, 'B': 0.686, 'C': 0.73,
    'D': 0.77, 'E': 0.632, 'F': 0.59, 'G': 0.775, 'H': 0.752, 'I': 0.295, 'J': 0.571, 'K': 0.672, 'L': 0.565,
    'M': 0.903, 'N': 0.753, 'O': 0.787, 'P': 0.639, 'Q': 0.787, 'R': 0.695, 'S': 0.642, 'T': 0.646, 'U': 0.744,
    'V': 0.69, 'W': 0.989, 'X': 0.685, 'Y': 0.679, 'Z': 0.685, '[': 0.39, '\\': 0.36, ']': 0.39, '^': 0.838,
    '_': 0.5, '`': 0.5, 'a': 0.613, 'b': 0.635, 'c': 0.571, 'd': 0.635, 'e': 0.615, 'f': 0.37, 'g': 0.635,
    'h': 0.634, 'i': 0.278, 'j': 0.278, 'k': 0.579, 'l': 0.278, 'm': 0.974, 'n': 0.634, 'o': 0.612, 'p': 0.635,
    'q': 0.635, 'r': 0.411, 's': 0.528, 't': 0.392, 'u': 0.634, 'v': 0.592, 'w': 0.818, 'x': 0.592, 'y': 0.592,
    'z': 0.552, '{': 0.636, '|': 0.337, '}': 0.636, '~': 0.838, '°': 0.5, '±': 0.838, '·': 0.318, '×': 0.838,
    'ĝ': 0.635, 'ĥ': 0.634, 'ŷ': 0.592, '̂': 0.873, '̃': 0.873, '̄': 0.873, 'Δ': 0.684, 'Π': 0.752, 'Σ': 0.689,
    'Ω': 0.77, 'α': 0.664, 'β': 0.638, 'γ': 0.592, 'δ': 0.612, 'ε': 0.541, 'ζ': 0.544, 'η': 0.634, 'θ': 0.621,
    'ι': 0.338, 'κ': 0.589, 'λ': 0.642, 'μ': 0.636, 'ν': 0.587, 'ξ': 0.559, 'π': 0.605, 'ρ': 0.635, 'σ': 0.634,
    'τ': 0.602, 'υ': 0.591, 'φ': 0.721, 'χ': 0.578, 'ψ': 0.778, 'ω': 0.837, '‑': 0.361, '–': 0.5, '—': 1.0,
    '‘': 0.318, '’': 0.318, '“': 0.518, '”': 0.518, '†': 0.5, '‡': 0.5, '…': 1.0, '′': 0.227, '″': 0.442,
    'ℓ': 0.444, 'ℝ': 0.792, '←': 0.954, '→': 0.954, '↔': 1.341, '∀': 0.684, '∂': 0.634, '∃': 0.656, '∇': 0.669,
    '∈': 0.871, '∑': 0.674, '−': 0.838, '√': 0.637, '∞': 0.936, '∩': 0.732, '∪': 0.732, '≈': 0.838, '≠': 0.838,
    '≤': 0.838, '≥': 0.838, '⋯': 1.0, '⌈': 0.656, '⌉': 0.656, '⟨': 0.656, '⟩': 0.656,
}
_ADVANCE_BOLD = {
    ' ': 0.348, '!': 0.456, '"': 0.523, '#': 0.838, '$': 0.696, '%': 1.004, '&': 0.872, "'": 0.326, '(': 0.457,
    ')': 0.457, '*': 0.54, '+': 0.838, ',': 0.38, '-': 0.465, '.': 0.38, '/': 0.379, '0': 0.696, '1': 0.696,
    '2': 0.696, '3': 0.696, '4': 0.696, '5': 0.696, '6': 0.696, '7': 0.696, '8': 0.696, '9': 0.696, ':': 0.4,
    ';': 0.4, '<': 0.838, '=': 0.838, '>': 0.838, '?': 0.58, '@': 1.0, 'A': 0.774, 'B': 0.762, 'C': 0.737,
    'D': 0.83, 'E': 0.683, 'F': 0.683, 'G': 0.821, 'H': 0.837, 'I': 0.372, 'J': 0.58, 'K': 0.775, 'L': 0.637,
    'M': 0.995, 'N': 0.837, 'O': 0.85, 'P': 0.733, 'Q': 0.85, 'R': 0.77, 'S': 0.72, 'T': 0.682, 'U': 0.812,
    'V': 0.774, 'W': 1.103, 'X': 0.771, 'Y': 0.724, 'Z': 0.725, '[': 0.457, '\\': 0.379, ']': 0.457, '^': 0.838,
    '_': 0.5, '`': 0.5, 'a': 0.675, 'b': 0.716, 'c': 0.593, 'd': 0.716, 'e': 0.678, 'f': 0.435, 'g': 0.716,
    'h': 0.712, 'i': 0.343, 'j': 0.343, 'k': 0.665, 'l': 0.343, 'm': 1.042, 'n': 0.712, 'o': 0.687, 'p': 0.716,
    'q': 0.716, 'r': 0.493, 's': 0.595, 't': 0.478, 'u': 0.712, 'v': 0.652, 'w': 0.924, 'x': 0.645, 'y': 0.652,
    'z': 0.582, '{': 0.712, '|': 0.365, '}': 0.712, '~': 0.838, '°': 0.5, '±': 0.838, '·': 0.38, '×': 0.838,
    'ĝ': 0.716, 'ĥ': 0.712, 'ŷ': 0.652, '̂': 0.873, '̃': 0.873, '̄': 0.873, 'Δ': 0.774, 'Π': 0.837, 'Σ': 0.692,
    'Ω': 0.85, 'α': 0.687, 'β': 0.716, 'γ': 0.681, 'δ': 0.687, 'ε': 0.557, 'ζ': 0.591, 'η': 0.712, 'θ': 0.687,
    'ι': 0.39, 'κ': 0.71, 'λ': 0.669, 'μ': 0.736, 'ν': 0.681, 'ξ': 0.591, 'π': 0.791, 'ρ': 0.716, 'σ': 0.779,
    'τ': 0.638, 'υ': 0.675, 'φ': 0.782, 'χ': 0.645, 'ψ': 0.813, 'ω': 0.869, '‑': 0.422, '–': 0.5, '—': 1.0,
    '‘': 0.38, '’': 0.38, '“': 0.657, '”': 0.657, '†': 0.552, '‡': 0.552, '…': 1.0, '′': 0.264, '″': 0.512,
    'ℓ': 0.487, 'ℝ': 0.801, '←': 0.954, '→': 0.954, '↔': 1.341, '∀': 0.774, '∂': 0.658, '∃': 0.683, '∇': 0.697,
    '∈': 0.896, '∑': 0.718, '−': 0.838, '√': 0.667, '∞': 0.966, '∩': 0.812, '∪': 0.812, '≈': 0.838, '≠': 0.838,
    '≤': 0.838, '≥': 0.838, '⋯': 1.0, '⌈': 0.656, '⌉': 0.656, '⟨': 0.656, '⟩': 0.656,
}


class Measurer:
    """Sum of glyph advances; combining marks count as full glyphs, which only adds room."""

    def width(self, text: str, size: float, weight: str = "normal", family: str = FONT) -> float:
        plain = strip_markup(text)
        if family == MONO:
            return 0.62 * size * len(plain)
        table = _ADVANCE_BOLD if weight in {"bold", "600", "700"} else _ADVANCE_NORMAL
        return sum(table.get(ch, 0.7) for ch in plain) * size * 1.02


MEASURE = Measurer()
_MARKUP = re.compile(r"([_^])\{([^}]*)\}")


def strip_markup(text: str) -> str:
    return _MARKUP.sub(lambda m: m.group(2), text)


def render_markup(text: str, size: float = 14) -> str:
    """Use explicit sizes and baseline offsets supported by browsers and CairoSVG."""
    out = []
    pos, offset = 0, 0.0
    for m in _MARKUP.finditer(text):
        plain = text[pos:m.start()]
        if plain:
            out.append(f'<tspan dy="{-offset:g}" font-size="{size:g}">{escape(plain)}</tspan>')
            offset = 0.0
        shift = size * (0.22 if m.group(1) == "_" else -0.35)
        out.append(f'<tspan dy="{shift - offset:g}" font-size="{size * 0.72:g}">{escape(m.group(2))}</tspan>')
        offset = shift
        pos = m.end()
    if text[pos:]:
        out.append(f'<tspan dy="{-offset:g}" font-size="{size:g}">{escape(text[pos:])}</tspan>')
    return "".join(out)


def wrap(text: str, size: float, max_width: float, weight: str = "normal", family: str = FONT) -> list[str]:
    """Greedy word wrap on measured widths; an explicit newline forces a break."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = word if not current else current + " " + word
            if MEASURE.width(candidate, size, weight, family) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
class Canvas:
    def __init__(self, width: int, height: int, title: str, desc: str) -> None:
        self.w, self.h = width, height
        self.title, self.desc = title, desc
        self.parts: list[str] = []
        self.markers: dict[str, str] = {}
        self.overflows: list[str] = []
        self.nodes: dict[str, tuple[float, float, float, float]] = {}
        self.edges: list[dict[str, str]] = []
        self.contract: dict = {}

    # --- primitives -------------------------------------------------------
    def rect(self, x: float, y: float, w: float, h: float, *, fill: str = "#ffffff", stroke: str | None = BORDER,
             sw: float = 1.0, rx: float = 8.0, dash: str | None = None) -> None:
        attrs = [f'x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"']
        attrs.append(f'stroke="{stroke}" stroke-width="{sw}"' if stroke else 'stroke="none"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f"<rect {' '.join(attrs)}/>")

    def circle(self, cx: float, cy: float, r: float, *, fill: str, stroke: str | None = None, sw: float = 1.0) -> None:
        s = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else 'stroke="none"'
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" {s}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = LINE, sw: float = 1.2,
             dash: str | None = None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
                          f'stroke-width="{sw}" stroke-linecap="round"{d}/>')

    def _marker(self, color: str) -> str:
        key = "arrow-" + color.lstrip("#")
        if key not in self.markers:
            self.markers[key] = (f'<marker id="{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                                 f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
                                 f'fill="{color}"/></marker>')
        return key

    def arrow(self, points: list[tuple[float, float]], *, color: str = LINE, sw: float = 1.8,
              dash: str | None = None, head: bool = True, tail: bool = False, radius: float = 10.0) -> None:
        """Polyline arrow with rounded corners; points are the corner coordinates."""
        points = [point for i, point in enumerate(points) if i == 0 or point != points[i - 1]]
        if len(points) < 2:
            raise ValueError("An arrow needs two distinct points")
        d = self._rounded_path(points, radius)
        attrs = [f'd="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" '
                 f'stroke-linecap="round"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if head:
            attrs.append(f'marker-end="url(#{self._marker(color)})"')
        if tail:
            attrs.append(f'marker-start="url(#{self._marker(color)})"')
        self.parts.append(f"<path {' '.join(attrs)}/>")

    @staticmethod
    def _rounded_path(points: list[tuple[float, float]], radius: float) -> str:
        if len(points) < 3 or radius <= 0:
            return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        d = [f"M {points[0][0]:.1f},{points[0][1]:.1f}"]
        for i in range(1, len(points) - 1):
            (x0, y0), (x1, y1), (x2, y2) = points[i - 1], points[i], points[i + 1]
            l1 = math.hypot(x1 - x0, y1 - y0)
            l2 = math.hypot(x2 - x1, y2 - y1)
            r = min(radius, l1 / 2, l2 / 2)
            ax, ay = x1 - (x1 - x0) / l1 * r, y1 - (y1 - y0) / l1 * r
            bx, by = x1 + (x2 - x1) / l2 * r, y1 + (y2 - y1) / l2 * r
            d.append(f"L {ax:.1f},{ay:.1f} Q {x1:.1f},{y1:.1f} {bx:.1f},{by:.1f}")
        d.append(f"L {points[-1][0]:.1f},{points[-1][1]:.1f}")
        return " ".join(d)

    def text(self, x: float, y: float, s: str, *, size: float = 14, weight: str = "normal", fill: str = INK,
             anchor: str = "start", italic: bool = False, family: str = FONT, spacing: float | None = None,
             max_width: float | None = None) -> None:
        width = MEASURE.width(s, size, weight, family)
        if max_width is not None:
            if width > max_width + 0.5:
                self.overflows.append(f"{self.title}: '{strip_markup(s)}' needs {width:.0f}px, has {max_width:.0f}px")
        left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
        if left < -0.5 or left + width > self.w + 0.5 or y - size < 0 or y + size > self.h:
            self.overflows.append(f"{self.title}: text outside canvas: {strip_markup(s)!r}")
        style = f'font-family="{family}" font-size="{size:g}" fill="{fill}"'
        if weight != "normal":
            style += f' font-weight="{weight}"'
        if italic:
            style += ' font-style="italic"'
        if spacing:
            style += f' letter-spacing="{spacing}"'
        # Some SVG rasterizers anchor each tspan independently. Position mixed
        # text from its left edge so subscript runs cannot overlap one another.
        if anchor != "start" and _MARKUP.search(s):
            x, anchor = left, "start"
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
                          f'{style}>{render_markup(s, size)}</text>')

    def lines(self, x: float, y: float, items: list[str], *, size: float = 13, lh: float | None = None,
              fill: str = INK, anchor: str = "start", weight: str = "normal", family: str = FONT,
              max_width: float | None = None, italic: bool = False) -> float:
        """Draw consecutive lines, the first centred on ``y``; return the height used."""
        lh = lh or round(size * 1.38)
        for i, item in enumerate(items):
            self.text(x, y + i * lh, item, size=size, fill=fill, anchor=anchor, weight=weight, family=family,
                      max_width=max_width, italic=italic)
        return lh * len(items)

    def paragraph(self, x: float, y: float, text: str, max_width: float, *, size: float = 13,
                  lh: float | None = None, fill: str = INK, anchor: str = "start", weight: str = "normal",
                  italic: bool = False, family: str = FONT) -> float:
        return self.lines(x, y, wrap(text, size, max_width, weight, family), size=size, lh=lh, fill=fill,
                          anchor=anchor, weight=weight, italic=italic, family=family, max_width=max_width)

    # --- composites -------------------------------------------------------
    def panel(self, x: float, y: float, w: float, h: float, label: str, color: tuple[str, str, str],
              *, label_size: float = 12) -> None:
        self.rect(x, y, w, h, fill=color[1], stroke=color[2], sw=1.2, rx=14)
        self.rect(x, y, 6, h, fill=color[0], stroke=None, rx=3)
        self.text(x + 18, y + 17, label, size=label_size, weight="600", fill=color[0], spacing=0.8,
                  max_width=w - 30)

    def card(self, x: float, y: float, w: float, h: float, title: str, body: list[str] | str,
             color: tuple[str, str, str] = ART, *, title_size: float = 14, body_size: float = 12,
             fill: str | None = None, align: str = "start", accent: bool = True, lh: float | None = None,
             body_fill: str = INK, title_fill: str | None = None) -> None:
        self.rect(x, y, w, h, fill=fill or "#ffffff", stroke=color[2], sw=1.2, rx=9)
        if accent:
            self.rect(x, y, w, 4, fill=color[0], stroke=None, rx=2)
        pad = 11
        tx = x + pad if align == "start" else x + w / 2
        anchor = "start" if align == "start" else "middle"
        ty = y + 10 + title_size * 0.9
        if title:
            self.text(tx, ty, title, size=title_size, weight="600", fill=title_fill or color[0], anchor=anchor,
                      max_width=w - 2 * pad)
            ty += title_size * 1.2 + 2
        items = body if isinstance(body, list) else [body]
        wrapped: list[str] = []
        for item in items:
            wrapped.extend(wrap(item, body_size, w - 2 * pad))
        lh = lh or round(body_size * 1.4)
        used = ty + len(wrapped) * lh - lh / 2 + 4
        if used > y + h + 0.5:
            self.overflows.append(f"{self.title}: card '{strip_markup(title)}' body needs {used - y:.0f}px, has {h:.0f}px")
        self.lines(tx, ty + body_size * 0.15, wrapped, size=body_size, lh=lh, fill=body_fill,
                   anchor=anchor, max_width=w - 2 * pad)

    def node(self, node_id: str, x: float, y: float, w: float, h: float,
             title: str, body: list[str] | str, color=ART, *, phase: str = "",
             title_size: float = 19, body_size: float = 17, **kwargs) -> None:
        """Named block: SVG IDs and edge endpoints make the drawing inspectable."""
        if node_id in self.nodes:
            raise ValueError(f"Duplicate diagram node {node_id}")
        self.nodes[node_id] = (x, y, w, h)
        self.parts.append(f'<g id="{escape(node_id)}" data-phase="{escape(phase)}">')
        self.card(x, y, w, h, title, body, color, title_size=title_size,
                  body_size=body_size, **kwargs)
        self.parts.append('</g>')

    def edge(self, source: str, target: str, points: list[tuple[float, float]],
             *, role: str = "flow", **kwargs) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Unknown edge endpoint {source} -> {target}")
        record = {"source": source, "target": target, "role": role}
        self.edges.append(record)
        self.arrow(points, **kwargs)
        attrs = " ".join(f'data-{k}="{escape(v)}"' for k, v in record.items())
        self.parts[-1] = self.parts[-1].replace('<path ', f'<path {attrs} ', 1)

    def pill(self, x: float, y: float, label: str, color: tuple[str, str, str], *, size: float = 11,
             anchor: str = "start", weight: str = "600") -> float:
        w = MEASURE.width(label, size, weight) + 16
        x0 = x if anchor == "start" else (x - w / 2 if anchor == "middle" else x - w)
        self.rect(x0, y - size * 0.85, w, size * 1.7, fill=color[1], stroke=color[2], sw=1, rx=size * 0.85)
        self.text(x0 + w / 2, y, label, size=size, weight=weight, fill=color[0], anchor="middle")
        return w

    def legend(self, x: float, y: float, items=LEGEND, *, size: float = 12, gap: float = 22) -> None:
        cx = x
        for label, color in items:
            self.rect(cx, y - 7, 14, 14, fill=color[1], stroke=color[0], sw=1.4, rx=3)
            self.text(cx + 20, y, label, size=size, fill=MUTED)
            cx += 20 + MEASURE.width(label, size) + gap

    def title_block(self, title: str, subtitle: str) -> None:
        self.text(40, 44, title, size=28, weight="700", fill=INK, max_width=self.w - 80)
        self.text(40, 79, subtitle, size=17, fill=MUTED, max_width=self.w - 80)

    def render(self) -> str:
        defs = "".join(self.markers.values())
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="title desc">\n'
                f'<title id="title">{escape(self.title)}</title>\n<desc id="desc">{escape(self.desc)}</desc>\n'
                f'<metadata id="diagram-contract">{escape(json.dumps(self.contract, sort_keys=True))}</metadata>\n'
                f'<defs>{defs}</defs>\n'
                f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="18" fill="#ffffff" '
                f'stroke="{BORDER}" stroke-width="1"/>\n')
        return head + "\n".join(self.parts) + "\n</svg>\n"


# ---------------------------------------------------------------------------
def load_registry() -> tuple[dict, dict]:
    constants = json.loads((PAPER / "registered_constants.json").read_text(encoding="utf-8"))
    results = json.loads((PAPER / "reported_results.json").read_text(encoding="utf-8"))
    return constants, results


def fmt(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def pct(value: float) -> str:
    return f"{round(value * 100):d}%"


# ---------------------------------------------------------------------------
def figure_architecture(C: dict, R: dict) -> Canvas:
    obs = C["observer_eq12_table_S9"]
    cv = Canvas(1080, 856, "Frozen-plant control loop",
                "A feedback episode reads hidden states and frozen-head predictions without labels. "
                "Measurements enter a conditioned observer and a supported action policy. Ordinary actions "
                "run the registered native or inserted site; A8 and A9 directly restore a reference state. "
                "Both paths update the hidden state for the next depth. Observer and consequence models "
                "are fitted offline and remain fixed. Global comparator admission is a separate workflow.")
    cv.contract = {"kind": "feedback_execution", "labels_enter_runtime": False,
                   "parameter_updates_online": False,
                   "ordinary_actions": list(range(8)), "reset_actions": [8, 9],
                   "global_admission_in_loop": False, "open_loop_observation_updates": False}
    cv.title_block("A control loop around frozen message passing",
                   "Feedback changes transport or restores a state. Predictor and controller parameters stay fixed.")
    cv.panel(32, 108, 1016, 204, "FROZEN PREDICTOR · one complete graph per episode", PLANT, label_size=16)
    cv.node("input", 52, 158, 152, 104, "Graph input", ["features X", "operator P"], ART, phase="runtime")
    cv.node("encoder", 234, 158, 184, 104, "Encoder", ["frozen weights", "source H_{0}"], PLANT, phase="runtime")
    cv.node("state", 448, 158, 184, 104, "State H_{ℓ}", ["starts at H_{0}", "then evolves"], PLANT, phase="runtime")
    cv.node("head", 674, 158, 184, 104, "Task head", ["same frozen head", "local readout"], PLANT,
            phase="runtime", body_size=16)
    cv.node("predictions", 900, 158, 128, 104, "p_{ℓ}", ["confidence", "final at stop"], PLANT,
            phase="runtime", title_size=21, body_size=16)
    for source, target, x1, x2 in [("input", "encoder", 204, 234), ("encoder", "state", 418, 448),
                                  ("state", "head", 632, 674), ("head", "predictions", 858, 900)]:
        cv.edge(source, target, [(x1, 210), (x2, 210)], color=PLANT[0], sw=2)
    cv.lines(52, 281, ["P is nonnegative and state-independent.",
                      "It is row-normalized; native weights stay fixed."],
             size=16, lh=21, fill=MUTED, max_width=440)

    cv.panel(32, 388, 1016, 192, "LABEL-FREE FEEDBACK · no labels enter this loop", SENSE, label_size=16)
    cv.node("measure", 52, 436, 280, 118, "Measure y_{ℓ}",
            ["confidence + state geometry", "cross-spectral (Full profile)", "depth + transport history"], SENSE,
            phase="runtime")
    cv.node("observer", 370, 436, 280, 118, "Estimate state s_{ℓ}",
            [f"{obs['state_dimension']}-state conditioned model", "covariance + innovations",
             f"correction every {obs['observer_period']} depths"], SENSE, phase="runtime", body_size=16)
    cv.node("policy", 688, 436, 340, 118, "Choose action a_{ℓ}",
            ["support-qualified critic / MPC", "one-step · radius · MPC-RH", "internal recovery when needed"], POLICY,
            phase="runtime")
    cv.edge("state", "measure", [(540, 262), (540, 346), (192, 346), (192, 436)],
            role="measurement", color=SENSE[0], sw=2)
    cv.edge("predictions", "measure", [(964, 262), (964, 346), (192, 346), (192, 436)],
            role="measurement", color=SENSE[0], sw=2)
    cv.circle(540, 346, 3.5, fill=SENSE[0])
    cv.text(402, 368, "hidden states and predictions", size=17, fill=SENSE[0])
    cv.edge("measure", "observer", [(332, 495), (370, 495)], color=SENSE[0], sw=2)
    cv.edge("observer", "policy", [(650, 495), (688, 495)], color=SENSE[0], sw=2)

    cv.node("offline_models", 52, 638, 350, 128, "Fitted offline, then frozen",
            ["D_{explore}: observer, critic and", "sequence models; separate", "empirical-radius holdout"], MODEL,
            phase="offline", body_size=17)
    cv.node("ordinary", 450, 638, 282, 128, "A0–A7 · site dispatch",
            ["native aggregation, or", "inserted transport", "A0 preserves the nominal law"], POLICY,
            phase="runtime", body_size=16)
    cv.node("reset", 754, 638, 274, 128, "A8 / A9 · direct reset",
            ["source H_{0} / saved H_{safe}", "no learned update", "anchor depth is recorded"], POLICY,
            phase="runtime", body_size=16)
    cv.edge("offline_models", "observer", [(227, 638), (227, 612), (510, 612), (510, 554)],
            role="frozen_parameters", color=SENSE[0], dash="5 5", sw=1.7)
    cv.edge("policy", "ordinary", [(858, 554), (858, 604), (591, 604), (591, 638)],
            role="ordinary_action", color=POLICY[0], sw=2)
    cv.edge("policy", "reset", [(858, 554), (858, 604), (891, 604), (891, 638)],
            role="reset_action", color=POLICY[0], sw=2)
    cv.circle(858, 604, 3.5, fill=POLICY[0])
    # The return path is outside all blocks; it enters the current state from above.
    for source, x in (("ordinary", 591), ("reset", 891)):
        cv.edge(source, "state", [(x, 766), (x, 792), (1060, 792), (1060, 144),
                                  (540, 144), (540, 158)],
                role="next_state", color=PLANT[0], sw=1.8)
    cv.text(650, 813, "H_{ℓ+1} → next control depth", size=17, fill=PLANT[0])
    cv.text(40, 836, "Open-loop schedules keep their plan. Global admission and comparator return are separate decisions.",
            size=16, fill=MUTED, max_width=1000)
    return cv


def figure_lifecycle(C: dict, R: dict) -> Canvas:
    roles = C["data_roles_table_S2"]
    alloc = roles["official_training_allocation"]
    cv = Canvas(1080, 1200, "Separated qualification and evaluation workflow",
                "Eight stages: prepare, train, explore, fit, select, confirm, admit and evaluate. "
                "Official training is divided 60/20/10/10 into fit, exploration, selection and confirmation roles. "
                "The candidate registry is frozen before selection. Confirmation fixes the candidate and comparator "
                "before official validation admission. Passing admission retains the executable candidate; "
                "failure returns the comparator. The final policy is locked before official test labels are used. "
                "This is the repository's separated sequence, not the reused-validation protocol of the reported results.")
    cv.contract = {"kind": "offline_lifecycle",
                   "stages": ["prepare", "train", "explore", "fit", "select", "confirm", "admit", "evaluate"],
                   "training_allocation": alloc, "registry_lock_before": "select",
                   "candidate_fixed_before": "admit", "policy_lock_before": "evaluate",
                   "admission_pass": "executable_candidate", "admission_fail": "comparator",
                   "sequence_scope": "separated repository execution"}
    cv.title_block("Qualify a policy before opening the test labels",
                   "Eight sealed stages · separated label roles · one final deployment decision")
    cv.text(42, 116, "STAGE", size=16, weight="700", fill=MUTED, spacing=1)
    cv.text(214, 116, "OPERATION AND OUTPUT", size=16, weight="700", fill=MUTED, spacing=1)
    cv.text(804, 116, "LABEL ROLE", size=16, weight="700", fill=MUTED, spacing=1)
    rows = [
        ("prepare", 140, "Prepare graphs and separated roles", "Graph identity · units · split manifest",
         "split assignment", "no fitted decisions", ART),
        ("train", 250, "Fit q_{0} and the nominal plant", "Freeze encoder, native blocks and task head",
         "D_{fit}", f"{pct(alloc['D_fit'])} of official training", PLANT),
        ("explore", 360, "Log transitions and paired action effects", "Frozen plant · behavior propensities · targets",
         "D_{explore}", f"{pct(alloc['D_explore'])} of official training", SENSE),
        ("fit", 470, "Fit the observer and consequence models", "Freeze learned models and empirical radii",
         "logged D_{explore}", "separate fitting groups", SENSE),
        ("select", 618, "Select comparator q_{C} and provisional q̂_{s}", "Nested comparison · simplest eligible candidate",
         "D_{select}", f"{pct(alloc['D_select'])}; entry uses D_{{explore}}", POLICY),
        ("confirm", 728, "Retain or downgrade the candidate q_{M}", "Required contrasts · eligible predecessor or q_{C}",
         "D_{confirm}", f"{pct(alloc['D_confirm'])} of official training", POLICY),
        ("admit", 838, "Compare the fixed executable rule with q_{C}", "Loss + metric gates choose the final rule q_{D}",
         "D_{admit}", "official validation", POLICY),
        ("evaluate", 1080, "Report q_{0}, q_{N}, q_{C} and q_{D}", "Task metrics + paired effects; no policy changes",
         "D_{test}", "official test", ART),
    ]
    for i, (name, y, title, body, role, note, color) in enumerate(rows):
        cv.rect(40, y, 136, 88, fill=color[1], stroke=color[2], sw=1.3, rx=9)
        cv.text(54, y + 23, f"{i + 1:02d}", size=17, weight="600", fill=color[0])
        cv.text(54, y + 57, name, size=20, weight="700", fill=INK)
        cv.node(name, 202, y, 836, 88, title, body, color, phase="offline", title_size=19,
                body_size=17, accent=False)
        # Data roles occupy their own column, without arrows into runtime inference.
        cv.rect(790, y + 8, 236, 72, fill=color[1], stroke=None, rx=6)
        cv.text(802, y + 30, role, size=18, weight="600", fill=color[0], max_width=212)
        cv.text(802, y + 59, note, size=15.5, fill=MUTED, max_width=212)
        # Keep the operation text to the left of the role column.
        if MEASURE.width(title, 19, "600") > 566:
            cv.overflows.append(f"Lifecycle operation title overlaps label column: {strip_markup(title)}")
        if MEASURE.width(body, 17) > 566:
            cv.overflows.append(f"Lifecycle operation body overlaps label column: {strip_markup(body)}")
    for source, target, end, start in [(rows[i][0], rows[i + 1][0], rows[i][1] + 88, rows[i + 1][1])
                                      for i in range(6)]:
        cv.edge(source, target, [(108, end), (108, start)], color=LINE, sw=1.7)
    cv.line(202, 589, 1038, 589, color=SENSE[0], dash="5 5")
    cv.pill(620, 589, "CANDIDATE REGISTRY FROZEN BEFORE SELECTION", SENSE, size=15, anchor="middle")

    cv.node("retain_candidate", 270, 960, 304, 62, "PASS → retain q̃_{M}", [], SENSE,
            phase="offline", title_size=19, accent=False)
    cv.node("return_comparator", 672, 960, 304, 62, "FAIL → return q_{C}", [], ART,
            phase="offline", title_size=19, accent=False)
    cv.edge("admit", "retain_candidate", [(566, 926), (566, 942), (422, 942), (422, 960)],
            role="admission_pass", color=SENSE[0], sw=1.8)
    cv.edge("admit", "return_comparator", [(566, 926), (566, 942), (824, 942), (824, 960)],
            role="admission_fail", color=ART[0], sw=1.8)
    cv.circle(566, 942, 3.3, fill=POLICY[0])
    for source, x in (("retain_candidate", 422), ("return_comparator", 824)):
        cv.edge(source, "evaluate", [(x, 1022), (x, 1054), (620, 1054), (620, 1080)],
                role="locked_policy", color=PLANT[0], sw=1.7)
    cv.pill(620, 1054, "POLICY LOCK · FINAL RULE FIXED BEFORE TEST", PLANT, size=15.5, anchor="middle")
    cv.text(40, 1183, "Separated execution shown. The paper's reused-validation admission statistics remain descriptive.",
            size=16, fill=MUTED, max_width=1000)
    return cv


def figure_control_depth(C: dict, R: dict) -> Canvas:
    reg = C["action_registry_table_S6"]
    actions, derived = reg["actions"], reg["derived_table_S6b"]
    gps = C["backbones_table_S5"]["PascalVOC-SP"]["identity_test_E1"]
    peptide = C["backbones_table_S5"]["Peptides-func"]["identity_test_E1"]
    cv = Canvas(1080, 1116, "Transport actions and site semantics",
                "The common transport is a lazy mixture of a one-step operator and a mass-corrected truncated "
                "resolvent. Shared blocks use their frozen update. GPS and linear-aggregation GINE-style plants "
                "distinguish native aggregation sites from parameter-free inserted sites: A0 is native mean "
                "aggregation at the former and identity at the latter. A8 and A9 are direct state assignments. "
                "The action table gives the common-coordinate coefficients, expected hops, identity coefficient "
                "and exploration propensity; these values are read from the registered constants.")
    cv.contract = {"kind": "transport_semantics", "resolvent_order": 8,
                   "native_A0": "P", "inserted_A0": "I", "source_reference": "H0",
                   "direct_resets_bypass_update": True,
                   "gps_sites": {"native": gps["native_sites"], "inserted": gps["inserted_sites"]},
                   "peptide_sites": {"native": peptide["native_sites"], "inserted": peptide["inserted_sites"]},
                   "table_scope": "common transport before source retention"}
    cv.title_block("Transport actions and the site they act on",
                   "Ordinary transport and direct state restoration have different execution semantics.")
    cv.panel(32, 110, 1016, 182, "COMMON TRANSPORT COORDINATES · A0–A7", POLICY, label_size=16)
    cv.text(54, 166, "T_{a}H = (1 − ν_{a})H + ν_{a}[(1 − ρ_{a})PH + ρ_{a}R_{η,8}H]", size=23, fill=INK,
            max_width=970)
    cv.text(54, 207, "R_{η,8}H = η Σ_{t=0…8}(1 − η)^{t}P^{t}H + (1 − η)^{9}H", size=22, fill=INK,
            max_width=970)
    cv.text(54, 249, "Local kernel: P. The resolvent has row mass one and uses eight sparse products.",
            size=17, fill=MUTED, max_width=970)
    cv.text(54, 275, "Shared block: H^{+} = LN[(1 − α)φ(T_{a}HW + b) + αH_{0}]", size=18, fill=PLANT[0],
            max_width=970)

    cv.node("native_site", 32, 310, 498, 132, "Native site · GPS / linear-GINE",
            ["Designated A0 uses P; other ordinary actions", "supply (1 − α)T_{a}H + αH_{0} as the message.",
             "The block's remaining computation stays fixed."], PLANT, body_size=17)
    cv.node("inserted_site", 550, 310, 498, 132, "Inserted site · parameter-free",
            ["A0 is literal identity; A1–A7 apply", "(1 − ν)H + ν[(1 − α)K_{a}H + αH_{0}].",
             "K_{a} is non-lazy; ν is applied only once."], POLICY, body_size=17)
    cv.text(40, 466, f"GPS: {gps['native_sites']} native + {gps['inserted_sites']} inserted sites.  "
            f"Linear-GINE style: {peptide['native_sites']} native + {peptide['inserted_sites']} inserted sites.",
            size=17, fill=MUTED, max_width=1000)
    cv.rect(32, 488, 1016, 66, fill=ART[1], stroke=BORDER, sw=1.2, rx=9)
    cv.text(54, 512, "A8: H^{+} ← H_{0}", size=21, weight="600", fill=INK)
    cv.text(340, 512, "A9: H^{+} ← H_{safe}", size=21, weight="600", fill=INK)
    cv.text(54, 539, "Direct resets bypass the learned update. A9 is excluded from open-loop planning.",
            size=17, fill=MUTED, max_width=970)

    tx, ty = 44, 596
    widths = [52, 250, 80, 80, 80, 80, 116, 110, 100]
    xs = [tx]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)
    headers = ["ID", "Action", "ν", "ρ", "α", "η", "Hops", "Identity", "p"]
    cv.rect(32, ty - 20, 1016, 40, fill=POLICY[1], stroke=None, rx=6)
    for x, header in zip(xs, headers):
        cv.text(x, ty, header, size=17, weight="700", fill=POLICY[0])
    for i, action in enumerate(actions):
        y = ty + 43 + i * 38
        cv.parts.append(f'<g id="action-row-A{action["id"]}" data-action="{action["id"]}">')
        if i % 2 == 1:
            cv.rect(32, y - 19, 1016, 38, fill=ART[1], stroke=None, rx=0)
        cv.text(xs[0], y, f"A{action['id']}", size=17, weight="700")
        cv.text(xs[1], y, action["role"], size=17)
        d = derived.get(f"A{action['id']}")
        if d is not None:
            values = [fmt(action["nu"], 2), fmt(action["rho"], 2), fmt(action["alpha"], 2),
                      "—" if action["eta"] is None else fmt(action["eta"], 2),
                      fmt(d["executed_range_hops_per_step"], 3), fmt(d["executed_identity_coefficient"], 3)]
            for x, value in zip(xs[2:8], values):
                cv.text(x, y, value, size=17, family=MONO)
        else:
            cv.text(xs[2], y, "assign H_{0}" if action["id"] == 8 else "assign saved H_{safe}",
                    size=17, italic=True, fill=MUTED)
            cv.text(xs[6], y, "0", size=17, family=MONO)
            cv.text(xs[7], y, "—", size=17, family=MONO)
        cv.text(xs[8], y, fmt(action["exploration_probability"], 2), size=17, family=MONO)
        cv.parts.append('</g>')
    cv.text(40, 1037, "Hops = expected transported hops. Identity = coefficient of P^{0}, not the graph-return diagonal.",
            size=16, fill=MUTED, max_width=1000)
    cv.text(40, 1065, "Table values use common transport coordinates before retention; designated-site A0 differs above.",
            size=16, fill=MUTED, max_width=1000)
    cv.text(40, 1093, "Sources: main Eqs. 8–10; Supplementary Tables S5b, S6 and S6b. No predictive guarantee is implied.",
            size=16, fill=MUTED, max_width=1000)
    return cv


def figure_policy_families(C: dict, R: dict) -> Canvas:
    obs = C["observer_eq12_table_S9"]
    cv = Canvas(1080, 834, "Nine routes and their information contracts",
                "Six open-loop families commit a schedule or stopping depth using development data and, where "
                "declared, the initial input. Three feedback families repeatedly use the current label-free state. "
                "MPC-OL plans once and completes with nominal transport; MPC-RH replans and executes one action. "
                "Selection chooses the strongest registered open-loop comparator before comparing feedback. "
                "Confirmation and admission can return that comparator. This diagram contains no measured route outcomes.")
    open_loop = ["Nominal", "Best depth", "Learned stop", "Phase schedule", "Fixed sequence", "MPC-OL"]
    feedback = ["One-step", "Radius", "MPC-RH"]
    cv.contract = {"kind": "policy_information", "open_loop": open_loop, "feedback": feedback,
                   "MPC_OL": {"replans": False, "intermediate_observations_change_plan": False,
                              "tail": "nominal", "A9_allowed": False},
                   "MPC_RH": {"replans": True, "executes": "first_action"}, "contains_measured_results": False}
    cv.title_block("Nine routes, distinguished by information and timing",
                   "The same frozen plant is evaluated under different decision rules.")
    cv.panel(32, 112, 1016, 314, "OPEN LOOP · commit before rollout; the initial input may condition the plan", ART,
             label_size=16)
    cards = [
        ("Nominal", ["A0 at every site", "full nominal depth"]),
        ("Best depth", ["development-selected depth", "nominal transport until stop"]),
        ("Learned stop", ["initial measurements", "nearest-template stop depth"]),
        ("Phase schedule", ["registered action phases", "fixed switch boundaries"]),
        ("Fixed sequence", ["exploration-derived proposal", "execute the registered sequence"]),
        ("MPC-OL", ["plan once from the initial state", "nominal completion after horizon"]),
    ]
    for i, (name, body) in enumerate(cards):
        cv.node(f"ol_{i}", 52 + (i % 3) * 328, 162 + (i // 3) * 122, 308, 104, name, body, ART,
                body_size=16)
    cv.panel(32, 448, 1016, 206, "FEEDBACK · use the current label-free state at each decision", SENSE, label_size=16)
    fb = [("One-step", ["minimize the critic mean", "over supported actions"]),
          ("Radius", ["mean + κ scale + empirical radius", "plus the registered ρ penalty"]),
          ("MPC-RH", ["replan with consequence models", "execute the first action; observe"])]
    for i, (name, body) in enumerate(fb):
        cv.node(f"fb_{i}", 52 + i * 328, 500, 308, 104, name, body, SENSE, body_size=16)
    cv.text(52, 633, f"MPC settings: horizons {{{', '.join(str(h) for h in obs['horizons'])}}}; "
            f"beam {obs['beam_width']}; top-{obs['top_k']}; discount γ = {fmt(obs['gamma'], 2)}.",
            size=16, fill=MUTED, max_width=970)
    cv.node("comparator", 52, 704, 280, 80, "Select q_{C}", "strongest registered open loop", ART, body_size=16)
    cv.node("qualify", 400, 704, 280, 80, "Compare + confirm", "eligible feedback or q_{C}", POLICY, body_size=16)
    cv.node("deploy", 748, 704, 280, 80, "Admit and lock q_{D}", "candidate or exact q_{C} return", PLANT, body_size=16)
    cv.edge("comparator", "qualify", [(332, 744), (400, 744)], color=POLICY[0], sw=2)
    cv.edge("qualify", "deploy", [(680, 744), (748, 744)], color=POLICY[0], sw=2)
    cv.text(40, 815, "Profiles, horizons, coefficients and ensemble sizes are registered before selection. Sources: Eq. 15; Tables S9–S11.",
            size=15.5, fill=MUTED, max_width=1000)
    return cv


def figure_review_workflow(C: dict, R: dict) -> Canvas:
    cv = Canvas(1080, 788, "From a sealed run to reviewable evidence",
                "A completed run records resolved settings, stage manifests, numeric observations, policy decisions "
                "and checkpoint identities. Three review paths check different objects: numeric replay recomputes "
                "decisions from saved observations; checkpoint replay recomputes predictions from trusted frozen "
                "weights and the prepared graph; a portable review bundle checks numeric evidence without shipping "
                "weights or prepared data. New official-data experiments and manuscript registry values retain "
                "separate provenance. Record consistency alone does not establish original experiment chronology.")
    cv.contract = {"kind": "evidence_artifacts", "numeric_replay_runs_predictor": False,
                   "checkpoint_replay_requires": ["prepared_graph", "trusted_checkpoints"],
                   "portable_bundle_includes_checkpoints": False, "original_chronology_authenticated": False}
    cv.title_block("From a sealed run to reviewable evidence",
                   "Numeric replay, checkpoint replay and portable review answer different questions.")
    cv.node("run", 52, 124, 976, 108, "A completed, sealed run",
            ["resolved config + source identity · stage manifests · policy lock",
             "observed predictions + paired units · decisions + checkpoint identities"], ART,
            title_size=21, body_size=18)
    panels = [
        ("numeric", 52, "Recompute decisions", ["saved numeric observations", "and sealed decision records"],
         ["reproduce_results.py"], "Paired statistics + gate decisions", ART),
        ("checkpoint", 390, "Reexecute predictions", ["prepared graph", "and trusted frozen checkpoints"],
         ["evaluate_checkpoints.py"], "Fresh predictions + identity checks", PLANT),
        ("bundle", 728, "Export portable evidence", ["numeric records + manifests", "without graph data or weights"],
         ["build_review_bundle.py", "verify_review_bundle.py"], "A separately verifiable review bundle", SENSE),
    ]
    for node_id, x, title, body, commands, output, color in panels:
        cv.node(node_id, x, 300, 300, 228, title, body, color, title_size=18, body_size=17)
        cv.line(x + 12, 413, x + 288, 413, color=BORDER)
        cv.lines(x + 12, 440, commands, size=16, lh=27, fill=color[0], family=MONO, max_width=276)
        cv.node(node_id + "_output", x, 560, 300, 88, "Checks", output, color, title_size=17, body_size=16,
                accent=False)
        cv.edge("run", node_id, [(540, 232), (540, 266), (x + 150, 266), (x + 150, 300)],
                color=LINE, sw=1.8)
        cv.edge(node_id, node_id + "_output", [(x + 150, 528), (x + 150, 560)], color=color[0], sw=1.8)
    cv.circle(540, 266, 3.5, fill=ART[0])
    cv.rect(32, 678, 1016, 76, fill=ART[1], stroke=BORDER, sw=1, rx=9)
    cv.text(52, 703, "Paper registry: reported measurements and derived calculations.", size=17, fill=INK)
    cv.text(52, 732, "New run: its own data, frozen weights, decisions and measured costs. Compare with provenance intact.",
            size=16, fill=MUTED, max_width=970)
    cv.text(40, 771, "These checks verify records and execution; they do not authenticate the original experiments' chronology.",
            size=16, fill=MUTED, max_width=1000)
    return cv


def figure_evidence_chain(C: dict, R: dict) -> Canvas:
    cv = Canvas(1080, 1416, "From the source identity to the manuscript's tables",
                "Four stages connect the code to the manuscript. The runtime source, the registered configurations "
                "and the Arrhenius site profile define a run; one Slurm job executes the eight sealed stages and "
                "leaves a run directory whose stage manifests chain the SHA-256 of every file; the run-archive "
                "contract packs, verifies, deposits and ingests that directory into runs/index.json, and the "
                "experiments outside the eight stages enter as measurement-record sets traced value by value to "
                "their raw logs; the paper build reads only ingested or bundled runs and deposited record sets and "
                "writes the tables and figures. "
                "make verify runs every check on each push. The seven bundled synthetic runs travel the same "
                "chain under a synthetic banner. A sealed chain shows that a run's bytes are the bytes it wrote; "
                "who ran it, when, and whether it is the run behind a reported number are declared, not proven.")
    cv.contract = {"kind": "evidence_chain",
                   "stages": ["source_and_configuration", "execution", "run_archive_contract", "paper_build"],
                   "run_identity": ["source_sha256", "config_sha256"],
                   "archive_steps": ["archive-run", "deposit", "ingest-run"],
                   "record_steps": ["records-index", "records-check", "records-tables"],
                   "paper_outputs": {"tables": ["T1", "T2", "S4", "S12", "S12c", "S13", "S14", "S15", "S23", "S41", "M1"],
                                     "figures": ["F1", "F2", "F3", "F4", "F5", "F6"]},
                   "verify_targets": ["registry-check", "fingerprint", "audit-artifacts", "example-views-check",
                                      "example-paper-check", "manifests-check", "records-check", "availability-check",
                                      "runs-check-if-present", "audit"],
                   "chronology_authenticated": False, "result_binding_is_declared": True}
    cv.title_block("From the source identity to the manuscript's tables",
                   "One chain: sealed run → verified archive → ingested index → run-derived tables and figures")

    buses: list[tuple] = []

    def panel_row(y, node_h, label, color, nodes, *, body_size=15.5, title_size=18, phase="", parallel=()):
        h = node_h + 52
        cv.panel(40, y, 1000, h, label, color, label_size=15)
        n = len(nodes)
        gap = 26
        w = (1000 - 36 - gap * (n - 1)) / n
        ids = []
        for i, (node_id, title, body) in enumerate(nodes):
            x = 58 + i * (w + gap)
            cv.node(node_id, x, y + 36, w, node_h, title, body, color, phase=phase,
                    title_size=title_size, body_size=body_size)
            ids.append((node_id, x, w))
        for (a, xa, wa), (b, xb, wb) in zip(ids, ids[1:]):
            if b in parallel:      # a separate path in the same stage: no sequence arrow
                continue
            ya = y + 36 + node_h / 2
            cv.edge(a, b, [(xa + wa, ya), (xb, ya)], color=color[0], sw=1.7)
        return y + h

    # 1 -- what every run seals
    y = 104
    end = panel_row(y, 156, "1 · SOURCE AND CONFIGURATION · what every run seals", PLANT, [
        ("source", "Runtime source identity",
         ["cross_morse/*.py + run_cross_morse*.py → one SHA-256 (make fingerprint)",
          "sealed as source_sha256 in every run_state.json"]),
        ("configs", "Registered configurations",
         ["7 datasets × smoke / pilot / full: split rule, seeds, folds, margins ε_{R}, ε_{eq}, ε_{a}",
          "design recorded in manifests/registered/"]),
        ("site", "Arrhenius site profile",
         ["configs/arrhenius-site.env.example — GH200 partition, wall time, project paths",
          "allocation identifiers stay placeholders"]),
    ], phase="definition")
    buses.append((end, end + 42, "snapshot of source + resolved configuration + profile", PLANT[0], "configs", "launcher"))

    # 2 -- execution
    y = end + 42
    end = panel_row(y, 182, "2 · EXECUTION · one Slurm job per run (docs/ARRHENIUS.md)", SENSE, [
        ("launcher", "Launcher",
         ["run_cross_morse_<dataset>.sh submit full",
          "snapshots source + config into SUITE/source/; 1 GPU, 24 h, requeue, resume from the last sealed stage"]),
        ("stages", "Eight sealed stages",
         ["prepare · train · explore · fit · select · confirm · admit · evaluate",
          "each stage writes a manifest with the SHA-256 of its files; later stages verify earlier ones"]),
        ("run", "Completed run directory",
         ["prepared/ · train/ checkpoints · explore/ trajectories · select/ · confirm/ · admit/ · final/ predictions",
          "environment.json · policy_lock.json · run_state.json"]),
    ], phase="execution")
    buses.append((end, end + 42, "completed suite directory", SENSE[0], "run", "archive"))

    # 3 -- archive contract
    y = end + 42
    end = panel_row(y, 200, "3 · DEPOSIT · run archives (docs/RUN_ARCHIVES.md) and measurement records (records/)", POLICY, [
        ("archive", "make archive-run",
         ["verifies the stage chain, then packs the run",
          "ARCHIVE_MANIFEST.json: file hashes, source and config identities"]),
        ("deposit", "Deposit with a DOI",
         ["Zenodo or the institutional repository",
          "recorded with it: archive SHA-256, source identity, commit hash"]),
        ("ingest", "make ingest-run",
         ["verifies again → runs/<stem>/ and runs/index.json",
          "manifests/official/ extracted; AVAILABILITY.md regenerated"]),
        ("records", "make records-index",
         ["records/<set>/: raw logs, code, summary.csv",
          "every value re-read from its raw log; records/index.json"]),
    ], body_size=15, title_size=17, phase="archive", parallel=("records",))
    buses.append((end, end + 42, "verified archives and record sets (runs/, records/)", POLICY[0],
                  "ingest", "evidence"))

    # 4 -- paper build
    y = end + 42
    end = panel_row(y, 196, "4 · PAPER BUILD · make paper (paper/build/, scripts/records.py)", ART, [
        ("evidence", "Evidence records",
         ["one JSON per run: identities, roles, seeds, entry, candidates, decisions, effects, costs"]),
        ("tables", "Tables",
         ["T1 T2 S4 S12 S12c S13 S14 S15 S23 S41 M1; record tables from records/",
          ".tex / .md / .csv; unmeasured cells stay dashes"]),
        ("figures", "Figures",
         ["F1 Fig. 4d · F2 Fig. 5a · F3 Fig. 2c · F4 Fig. S14b · F5 Table S41 · F6 Fig. S9"]),
        ("compare", "make compare-runs",
         ["run value beside the transcribed manuscript value, at printed precision"]),
    ], body_size=15.5, title_size=17, phase="build")

    for y_from, y_to, label, color, source, target in buses:
        cv.edge(source, target, [(540, y_from), (540, y_to)], color=color, sw=2.0, role="handoff")
        cv.pill(560, (y_from + y_to) / 2, label, ART, size=15, anchor="start")

    # 5 -- checks
    y = end + 20
    cv.rect(40, y, 1000, 138, fill=PASS[1], stroke=PASS[2], sw=1.2, rx=14)
    cv.rect(40, y, 6, 138, fill=PASS[0], stroke=None, rx=3)
    cv.text(58, y + 18, "5 · CHECKED ON EVERY PUSH · make verify", size=15, weight="600", fill=PASS[0], spacing=0.8)
    used = cv.paragraph(58, y + 46, "registry-check · fingerprint · audit-artifacts · example-views-check · "
                        "example-paper-check · manifests-check · records-check · availability-check · runs-check · audit",
                        964, size=15.5, fill=INK)
    cv.paragraph(58, y + 46 + used + 6, "The seven bundled synthetic runs travel the same chain into "
                 "paper/generated/examples/ under a synthetic banner; official archives are built into "
                 "paper/generated/official/ and compared with the manuscript.", 964, size=15.5, fill=MUTED)
    cv.paragraph(40, y + 164, "A sealed chain shows that a run's bytes are the bytes it wrote. Who ran it, when, and "
                 "whether it is the run behind a reported number are declared in the result binding, not proven "
                 "by the hash.", 1000, size=15, fill=MUTED)
    return cv


def figure_decision_chain(C: dict, R: dict) -> Canvas:
    cv = Canvas(1080, 1500, "How a run's numbers become Table 1 and Table 2",
                "Seven steps turn a completed run into the cells of main Tables 1 and 2 and the supplementary "
                "ledgers. The candidate family is executed on the selection units; nested selection applies the "
                "registered reading of Eq. 19 with the eligibility threshold, the near-best band and the simplicity "
                "order; the confirmation map applies predeclared contrasts with practical margins under Holm; "
                "admission applies the clipped empirical-Bernstein bound of Eq. 27, a coverage statement only on "
                "independent graphs; the policy lock fixes the deployed rule before any test label is read; the "
                "test-stage effect is a paired whole-unit bootstrap; the cost ledger reports controller, "
                "comparator, expected and assurance costs. Each step names the sealed file it is read from and "
                "the table cell it fills. The serving-time fallback probability is not measured in this release, "
                "so the expected cost equals the controller cost.")
    cv.contract = {"kind": "decision_chain",
                   "steps": ["candidates", "select", "confirm", "admit", "lock", "evaluate", "cost"],
                   "files": {"candidates": ["select/candidates.json", "select/candidate_ledger.csv"],
                             "select": ["select/decision.json"],
                             "confirm": ["confirm/decision.json", "confirm/paired_contrasts.json"],
                             "admit": ["admit/decision.json"], "lock": ["policy_lock.json"],
                             "evaluate": ["final/results_summary.json", "final/test_predictions.npz"],
                             "cost": ["select/candidates.json", "stages/select.json"]},
                   "tables": ["T1", "T2", "S12", "S13", "S14", "S15", "S23", "S41"],
                   "figures": ["F1", "F2", "F3", "F5", "F6"],
                   "fallback_probability_measured": False, "policy_lock_before": "evaluate",
                   "coverage_statement_requires": "independent graphs unused by earlier decisions"}
    cv.title_block("How a run's numbers become Table 1 and Table 2",
                   "Seven decision statistics · the sealed file each is read from · the table cell it fills")
    cv.text(42, 116, "STEP", size=16, weight="700", fill=MUTED, spacing=1)
    cv.text(214, 116, "STATISTIC AND RULE", size=16, weight="700", fill=MUTED, spacing=1)
    cv.text(744, 116, "RUN FILE  →  TABLE CELL", size=16, weight="700", fill=MUTED, spacing=1)
    rows = [
        ("candidates", 142, "D_{select}", "Candidate family", POLICY,
         ["9 route families (6 open-loop, 3 feedback) × E ∈ {1, 2, 4, 8, 12} × Lean / Full. Every candidate is "
          "executed on the selection units: loss, metric, serial cost in nominal E = 1 units, wall seconds."],
         "select/candidates.json · select/candidate_ledger.csv",
         "S12 every row · S15 · S23 cost columns"),
        ("select", 176, "D_{select}", "Nested selection · Eq. (19)", POLICY,
         ["Paired max-t bootstrap over the selection units gives simultaneous [L_{s}, U_{s}] against the nested "
          "comparator. Eligible: U_{s} < −ε_{R}. Near-best: U_{s} ≤ U_{min} + ε_{eq}. Lexicographic simplicity "
          "picks the provisional q̂_{s}; q_{C} is the strongest cross-fitted open-loop route."],
         "select/decision.json",
         "T1 “exact q_{C} identity; provisional” · S12 “role” · F3"),
        ("confirm", 200, "D_{confirm}", "Confirmation map · Supplementary S11", POLICY,
         ["Predeclared contrasts for the provisional route (state destruction, radius vs one-step, MPC-RH vs OL, "
          "Full vs Lean): the point estimate must lie beyond its practical margin and the Holm-adjusted P over "
          "the family must be below α. A failed contrast downgrades to the eligible predecessor or to q_{C}; "
          "the survivor is q̃_{M}."],
         "confirm/decision.json · confirm/paired_contrasts.json",
         "T1 “confirmation output” · S14 · F6"),
        ("admit", 200, "D_{admit}", "Admission · Eq. (27), Theorem 2", GATE,
         ["Paired loss differences q̃_{M} − q_{C} on the admission units, clipped at B: U_{a} = D̄ + the "
          "empirical-Bernstein penalty; G_{admit} = 1[U_{a} ≤ −ε_{a}] together with the metric gate. A "
          "coverage statement only on J independent graphs that no earlier decision used; on dependent cores "
          "the same numbers are descriptive."],
         "admit/decision.json",
         "T1 “global admission (criterion)” · S13"),
        ("lock", 120, "no labels", "Policy lock", PLANT,
         ["q_{D} = q̃_{M} when G_{admit} = 1, otherwise q_{C} exactly: same checkpoints, same actions. Both "
          "hashes are recorded before any test label is read."],
         "policy_lock.json",
         "T2 “deployed rule, E”"),
        ("evaluate", 142, "D_{test}", "Test-stage effect", ART,
         ["q_{D} and q_{C} are executed once on the test units; paired whole-unit bootstrap of the loss (nats) "
          "and of the official metric (pp). Descriptive on dependent cores; conditional on the fixed trained "
          "models on graph collections."],
         "final/results_summary.json · final/test_predictions.npz",
         "T1 “final effect over q_{C}; scope” · T2 · F1 · F2"),
        ("cost", 164, "no labels", "Cost ledger · Eq. (32), Definition 2", POLICY,
         ["C_{qC}, C_{qD} from the candidate records. C_{expected} = (1 − p_{fb}) C_{qD} + p_{fb} C_{qC}; "
          "p_{fb} is not measured by this release, so C_{expected} = C_{qD}. C_{assure} = C_{qD} + C_{qC} "
          "for a deployed feedback rule. λ* = |Δ loss| / (C_{qD} − C_{qC}). Qualification hours = Σ stage "
          "durations."],
         "select/candidates.json · stages/<stage>.json",
         "T2 “C_{qC}”, “C_{expected}”, “C_{assure}”, “λ*”, “qualification” · S23 · S41 · F5"),
    ]
    y = 140
    positions = []
    for i, (name, h, role, title, color, body, files, cells) in enumerate(rows):
        if name == "evaluate":
            y += 36  # room for the policy-lock line
        positions.append((name, y, h))
        cv.rect(40, y, 136, h, fill=color[1], stroke=color[2], sw=1.3, rx=9)
        cv.text(54, y + 23, f"{i + 1:02d}", size=17, weight="600", fill=color[0])
        cv.text(52, y + 54, name, size=18, weight="700", fill=INK, max_width=116)
        cv.text(54, y + 84, role, size=15, fill=MUTED, max_width=112)
        cv.node(name, 202, y, 506, h, title, body, color, phase="decision", title_size=18.5,
                body_size=15.5, accent=False)
        cv.node(name + "_record", 732, y, 308, h, "", [], ART, phase="record", accent=False)
        used = cv.paragraph(743, y + 24, files, 286, size=15.5, fill=PLANT[0], weight="600")
        cv.paragraph(743, y + 24 + used + 8, cells, 286, size=15.5, fill=INK)
        cv.edge(name, name + "_record", [(708, y + 40), (732, y + 40)], color=color[0], sw=1.5, role="written_to")
        y += h + 18
    for (a, ya, ha), (b, yb, hb) in zip(positions, positions[1:]):
        cv.edge(a, b, [(108, ya + ha), (108, yb)], color=LINE, sw=1.7)
    lock_end = positions[4][1] + positions[4][2]
    line_y = lock_end + 27
    cv.line(202, line_y, 1040, line_y, color=PLANT[0], dash="5 5")
    cv.pill(620, line_y, "POLICY LOCK · NO TEST LABEL IS READ BEFORE THIS LINE", PLANT, size=15, anchor="middle")
    cv.paragraph(40, y + 6, "Every value is read from the sealed run; nothing is imputed. A quantity the released "
                 "runner does not measure is printed as a dash and named in the table note.", 1000, size=15,
                 fill=MUTED)
    return cv


FIGURES = {
    "architecture": figure_architecture,
    "lifecycle": figure_lifecycle,
    "control_depth": figure_control_depth,
    "policy_families": figure_policy_families,
    "review_workflow": figure_review_workflow,
    "evidence_chain": figure_evidence_chain,
    "decision_chain": figure_decision_chain,
}


# ---------------------------------------------------------------------------
def compact_png(png_path: Path) -> None:
    """Reduce a flat-colour rasterization to a 256-colour palette when Pillow is available."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return
    with Image.open(png_path) as original:
        image = original.convert("RGB")
        image.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                       dither=Image.Dither.NONE).save(png_path, optimize=True)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_SOURCE_KEY = b"cross_morse_svg_sha256"


def png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Read checked PNG chunks without requiring a plotting or image library."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file")
    position, chunks = len(PNG_SIGNATURE), []
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", data, position)[0]
        end = position + 12 + length
        if end > len(data):
            raise ValueError("Truncated PNG payload")
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        checksum = struct.unpack_from(">I", data, position + 8 + length)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != checksum:
            raise ValueError("Invalid PNG checksum")
        chunks.append((kind, payload))
        position = end
        if kind == b"IEND":
            if position != len(data):
                raise ValueError("Unexpected data after PNG end")
            break
    if not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise ValueError("Missing PNG header")
    if chunks[-1][0] != b"IEND" or not any(kind == b"IDAT" for kind, _ in chunks):
        raise ValueError("Incomplete PNG")
    return chunks


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def stamp_png_source(png_path: Path, svg: str) -> None:
    """Attach the source fingerprint; this detects stale fallbacks, not pixel equality."""
    chunks = png_chunks(png_path.read_bytes())
    digest = hashlib.sha256(svg.encode("utf-8")).hexdigest().encode("ascii")
    tagged = [PNG_SIGNATURE]
    for kind, payload in chunks:
        if kind == b"tEXt" and payload.startswith(PNG_SOURCE_KEY + b"\x00"):
            continue
        if kind == b"IEND":
            tagged.append(png_chunk(b"tEXt", PNG_SOURCE_KEY + b"\x00" + digest))
        tagged.append(png_chunk(kind, payload))
    png_path.write_bytes(b"".join(tagged))


def svg_dimensions(svg: str) -> tuple[int, int]:
    match = re.search(r'<svg\b[^>]*width="(\d+)" height="(\d+)"', svg)
    if match is None:
        raise ValueError("SVG requires explicit integer dimensions")
    return int(match.group(1)), int(match.group(2))


def png_matches_svg(png_path: Path, svg: str, scale: float = 2.0) -> bool:
    """Check PNG structure, dimensions and its source fingerprint using the stdlib."""
    try:
        chunks = png_chunks(png_path.read_bytes())
        expected = hashlib.sha256(svg.encode("utf-8")).hexdigest().encode("ascii")
        actual = [payload.split(b"\x00", 1)[1] for kind, payload in chunks
                  if kind == b"tEXt" and payload.startswith(PNG_SOURCE_KEY + b"\x00")]
        width, height = svg_dimensions(svg)
        dimensions = struct.unpack_from(">II", chunks[0][1])
        return dimensions == (round(width * scale), round(height * scale)) and actual == [expected]
    except (OSError, ValueError, struct.error):
        return False


def rasterize(svg_path: Path, png_path: Path, scale: float = 2.0) -> bool:
    """Replace a PNG only after successful rendering and source-fingerprint checks."""
    with tempfile.TemporaryDirectory(prefix="cross-morse-diagram-") as temporary:
        candidate = Path(temporary) / "render.png"
        if not _rasterize(svg_path, candidate, scale):
            return False
        try:
            compact_png(candidate)
            svg = svg_path.read_text(encoding="utf-8")
            stamp_png_source(candidate, svg)
            if not png_matches_svg(candidate, svg, scale):
                return False
            png_path.write_bytes(candidate.read_bytes())
        except (OSError, ValueError, struct.error):
            return False
    return True


def _rasterize(svg_path: Path, png_path: Path, scale: float) -> bool:
    try:
        import cairosvg  # type: ignore
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=scale)
        return True
    except Exception:
        pass
    svg = svg_path.read_text(encoding="utf-8")
    width, height = svg_dimensions(svg)
    inkscape = shutil.which("inkscape")
    if inkscape:
        try:
            # Inkscape can otherwise try to initialize a profile in a read-only
            # home directory. A task-local profile also avoids user settings.
            with tempfile.TemporaryDirectory(prefix="cross-morse-inkscape-") as profile:
                environment = os.environ.copy()
                environment["INKSCAPE_PROFILE_DIR"] = profile
                completed = subprocess.run(
                    [inkscape, str(svg_path.resolve()), "--export-type=png", "--export-area-page",
                     f"--export-filename={png_path.resolve()}", f"--export-width={round(width * scale)}",
                     f"--export-height={round(height * scale)}"],
                    capture_output=True, timeout=60, check=False, env=environment)
                if completed.returncode == 0 and png_path.is_file():
                    return True
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        html = ("<!doctype html><html><head><meta charset='utf-8'><style>html,body{margin:0;padding:0;background:#fff}"
                "svg{display:block}</style></head><body>" + svg + "</body></html>")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale)
            page = context.new_page()
            page.set_content(html)
            page.evaluate("document.fonts.ready")
            page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": width, "height": height})
            browser.close()
        return True
    except Exception:
        return False


def build(names: list[str]) -> dict[str, str]:
    constants, results = load_registry()
    rendered: dict[str, str] = {}
    problems: list[str] = []
    for name in names:
        canvas = FIGURES[name](constants, results)
        rendered[name] = canvas.render()
        problems.extend(canvas.overflows)
    if problems:
        raise SystemExit("text does not fit its box:\n  " + "\n  ".join(problems))
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--figure", choices=sorted(FIGURES), action="append",
                        help="draw only this figure (repeatable); default: all")
    parser.add_argument("--png", action="store_true",
                        help="also rasterize PNG copies at 2x; with --check, verify their source fingerprints")
    parser.add_argument("--check", action="store_true",
                        help="regenerate in memory and fail if any committed SVG differs")
    args = parser.parse_args(argv)
    names = args.figure or sorted(FIGURES)
    rendered = build(names)
    if args.check:
        stale = []
        for name in names:
            svg_path = args.output_dir / f"{name}.svg"
            try:
                matches = svg_path.read_bytes() == rendered[name].encode("utf-8")
            except OSError:
                matches = False
            if not matches:
                stale.append(f"{name}.svg")
            if args.png and not png_matches_svg(args.output_dir / f"{name}.png", rendered[name]):
                stale.append(f"{name}.png")
        if stale:
            print("missing or stale figures: " + ", ".join(stale))
            return 1
        suffix = " and PNG source fingerprints" if args.png else ""
        print(f"{len(names)} figure(s) match their committed SVG{suffix}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in names:
        svg_path = args.output_dir / f"{name}.svg"
        svg_path.write_bytes(rendered[name].encode("utf-8"))
        print(f"wrote {svg_path}")
        if args.png:
            png_path = args.output_dir / f"{name}.png"
            if rasterize(svg_path, png_path):
                print(f"wrote {png_path}")
            else:
                failures.append(name)
                print(f"PNG rendering failed for {name}; an installed CairoSVG, Inkscape or Playwright Chromium is required.",
                      file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
