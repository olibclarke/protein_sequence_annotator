#!/usr/bin/env python3
"""Render per-chain sequence annotation tracks from PDB/mmCIF files.

Dependencies:
- Python 3.9+
- gemmi
- mkdssp

Optional external tools:
- rsvg-convert or cairosvg: used to export PDF; otherwise SVG is kept as the
  fallback output

Typical usage:
- python3 protein_seq_annotator.py model.cif
- python3 protein_seq_annotator.py model.pdb --bfac
- python3 protein_seq_annotator.py model.cif --chain A --svg
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gemmi


# Basic residue and color lookups used throughout the pipeline.
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}

AF_COLORS = [
    (0.0, 50.0, "#ff7d45"),
    (50.0, 70.0, "#ffdb13"),
    (70.0, 90.0, "#65cbf3"),
    (90.0, 101.0, "#0053d6"),
]

DEFAULT_SS_COLORS = {"H": "#c0392b", "E": "#f39c12", "C": "#7f8c8d", "M": "#555555"}
MIN_SS_LENGTH = {"H": 4, "E": 3}


@dataclass
class ResidueInfo:
    index: int
    aa: str
    modeled: bool = False
    plddt: Optional[float] = None
    ss: str = "C"


@dataclass
class MetricSummary:
    label: str
    minimum: float
    mean: float
    maximum: float
    low_anchor: Optional[float] = None
    mid_anchor: Optional[float] = None
    high_anchor: Optional[float] = None


# Parse user options: input file, chain selection, layout, and color mode.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-chain sequence annotation SVG/PDF from PDB or mmCIF."
    )
    parser.add_argument("input", help="Input PDB or CIF/mmCIF file")
    parser.add_argument(
        "-o", "--output-dir", default=".", help="Directory for generated files"
    )
    parser.add_argument(
        "--chain",
        action="append",
        help="Chain ID to render. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--wrap", type=int, default=80, help="Residues per line in the output"
    )
    parser.add_argument(
        "--svg", action="store_true", help="Keep SVG output even when PDF export succeeds"
    )
    parser.add_argument(
        "--prefix",
        help="Output filename prefix. Defaults to the input stem.",
    )
    parser.add_argument(
        "--bfac",
        action="store_true",
        help="Color annotations by raw B-factor using a red-white-blue scale.",
    )
    return parser.parse_args()


def aa1(name: str) -> str:
    return AA3_TO_1.get(name.upper(), "X")


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value) or "chain"


def extract_chain_ids(raw_values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not raw_values:
        return None
    chain_ids: List[str] = []
    for raw in raw_values:
        for item in raw.split(","):
            item = item.strip()
            if item:
                chain_ids.append(item)
    return chain_ids or None


def is_polymer_chain(chain: gemmi.Chain) -> bool:
    try:
        return len(chain.get_polymer()) > 0
    except Exception:
        return False


def polymer_chains(model: gemmi.Model) -> List[gemmi.Chain]:
    return [chain for chain in model if is_polymer_chain(chain)]


# Map DSSP's detailed states onto the simplified helix/strand/coil track.
def dssp_code_to_track(code: str) -> str:
    if code in {"H", "G", "I", "P"}:
        return "H"
    if code in {"E", "B"}:
        return "E"
    return "C"


def run_dssp(input_path: Path, structure: gemmi.Structure) -> Dict[str, Dict[int, str]]:
    # Run DSSP on the original file when possible, or on a temporary PDB export
    # for mmCIF input, then keep only the per-residue chain/index assignments.
    mkdssp = shutil.which("mkdssp")
    if not mkdssp:
        raise RuntimeError("mkdssp is required for secondary-structure annotation")
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "annotated.dssp"
        dssp_input = input_path
        if input_path.suffix.lower() not in {".pdb", ".ent"}:
            dssp_input = Path(tmpdir) / "input.pdb"
            structure.write_pdb(str(dssp_input))
        subprocess.run(
            [mkdssp, str(dssp_input), str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assignments: Dict[str, Dict[int, str]] = {}
        seen_header = False
        for line in out.read_text(encoding="utf-8", errors="replace").splitlines():
            if not seen_header:
                if line.startswith("  #  RESIDUE AA STRUCTURE"):
                    seen_header = True
                continue
            if len(line) < 17 or line[13] == "!":
                continue
            chain_id = line[11].strip()
            if not chain_id:
                continue
            try:
                seq_num = int(line[5:10])
            except ValueError:
                continue
            assignments.setdefault(chain_id, {})[seq_num] = dssp_code_to_track(
                line[16].strip()
            )
        return assignments


def with_secondary_structure(
    input_path: Path, structure: gemmi.Structure
) -> tuple[gemmi.Structure, Dict[str, Dict[int, str]]]:
    # Always try DSSP so it can supplement files that have helices but no sheets.
    return structure, run_dssp(input_path, structure)


def find_chain(model: gemmi.Model, chain_id: str) -> gemmi.Chain:
    for chain in model:
        if chain.name == chain_id:
            return chain
    raise KeyError(f"chain {chain_id!r} not found")


def entity_sequence(structure: gemmi.Structure, chain: gemmi.Chain) -> List[str]:
    entity = structure.get_entity_of(chain.get_polymer())
    return [aa1(code) for code in entity.full_sequence]


# Sequence bookkeeping: align modeled residues onto the full polymer sequence.
def residue_position(residue: gemmi.Residue, sequence_len: int) -> Optional[int]:
    if residue.label_seq and 1 <= residue.label_seq <= sequence_len:
        return residue.label_seq
    if 1 <= residue.seqid.num <= sequence_len:
        return residue.seqid.num
    return None


def residue_plddt(residue: gemmi.Residue) -> Optional[float]:
    values = [atom.b_iso for atom in residue if atom.b_iso != 0]
    if not values:
        return None
    return sum(values) / len(values)


def residue_bfactor(residue: gemmi.Residue) -> Optional[float]:
    values = [atom.b_iso for atom in residue]
    if not values:
        return None
    return sum(values) / len(values)


def residue_records(
    structure: gemmi.Structure, chain: gemmi.Chain, use_bfactor: bool
) -> tuple[List[ResidueInfo], bool]:
    # Build a residue list for the full chain sequence, then mark which positions
    # are modeled and attach the active metric (pLDDT or raw B-factor).
    sequence = entity_sequence(structure, chain)
    residues = [ResidueInfo(index=i + 1, aa=aa) for i, aa in enumerate(sequence)]
    has_plddt = False
    for residue in chain.get_polymer():
        pos = residue_position(residue, len(residues))
        if pos is None:
            continue
        item = residues[pos - 1]
        item.modeled = True
        item.aa = aa1(residue.name)
        item.plddt = residue_bfactor(residue) if use_bfactor else residue_plddt(residue)
        has_plddt = has_plddt or item.plddt is not None
    return residues, has_plddt


def apply_ranges(
    residues: List[ResidueInfo], start: int, end: int, code: str
) -> None:
    for pos in range(max(1, start), min(len(residues), end) + 1):
        residues[pos - 1].ss = code


def enforce_min_ss_lengths(residues: List[ResidueInfo]) -> None:
    # Short DSSP/file-derived fragments are visually noisy, so collapse them back
    # to coil unless they meet the requested minimum lengths.
    start = 0
    while start < len(residues):
        code = residues[start].ss
        end = start
        while end + 1 < len(residues) and residues[end + 1].ss == code:
            end += 1
        min_len = MIN_SS_LENGTH.get(code)
        if min_len is not None and (end - start + 1) < min_len:
            for idx in range(start, end + 1):
                if residues[idx].modeled:
                    residues[idx].ss = "C"
        start = end + 1


# Merge DSSP-derived states with any explicit file annotations.
def assign_secondary_structure(
    structure: gemmi.Structure,
    chain: gemmi.Chain,
    residues: List[ResidueInfo],
    dssp_assignments: Optional[Dict[int, str]] = None,
) -> None:
    chain_name = chain.name
    if dssp_assignments:
        for seq_num, code in dssp_assignments.items():
            if 1 <= seq_num <= len(residues):
                residues[seq_num - 1].ss = code
    for helix in structure.helices:
        if helix.start.chain_name != chain_name or helix.end.chain_name != chain_name:
            continue
        apply_ranges(
            residues, helix.start.res_id.seqid.num, helix.end.res_id.seqid.num, "H"
        )
    for sheet in structure.sheets:
        for strand in sheet.strands:
            if strand.start.chain_name != chain_name or strand.end.chain_name != chain_name:
                continue
            apply_ranges(
                residues,
                strand.start.res_id.seqid.num,
                strand.end.res_id.seqid.num,
                "E",
            )
    for residue in residues:
        if not residue.modeled:
            residue.ss = "M"
    enforce_min_ss_lengths(residues)


def af_color(value: Optional[float], ss: str) -> str:
    if value is None:
        return DEFAULT_SS_COLORS.get(ss, DEFAULT_SS_COLORS["C"])
    for low, high, color in AF_COLORS:
        if low <= value < high:
            return color
    return AF_COLORS[-1][2]


def interpolate_rgb(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    rgb = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def bfactor_color(value: Optional[float], ss: str, metric_summary: Optional[MetricSummary]) -> str:
    if (
        value is None
        or metric_summary is None
        or metric_summary.low_anchor is None
        or metric_summary.mid_anchor is None
        or metric_summary.high_anchor is None
    ):
        return DEFAULT_SS_COLORS.get(ss, DEFAULT_SS_COLORS["C"])
    low = (0, 76, 255)
    mid = (229, 231, 235)
    high = (220, 0, 0)
    if value <= metric_summary.mid_anchor:
        denom = metric_summary.mid_anchor - metric_summary.low_anchor
        t = 0.0 if denom <= 0 else (value - metric_summary.low_anchor) / denom
        return interpolate_rgb(low, mid, max(0.0, min(1.0, t)))
    denom = metric_summary.high_anchor - metric_summary.mid_anchor
    t = 1.0 if denom <= 0 else (value - metric_summary.mid_anchor) / denom
    return interpolate_rgb(mid, high, max(0.0, min(1.0, t)))


def metric_color(
    value: Optional[float],
    ss: str,
    use_bfactor: bool,
    metric_summary: Optional[MetricSummary],
) -> str:
    if use_bfactor:
        return bfactor_color(value, ss, metric_summary)
    return af_color(value, ss)


def summarize_metric(
    residues: Sequence[ResidueInfo], use_bfactor: bool
) -> Optional[MetricSummary]:
    values = [residue.plddt for residue in residues if residue.plddt is not None]
    if not values:
        return None
    sorted_values = sorted(values)

    def percentile(fraction: float) -> float:
        index = max(0, min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1))))
        return sorted_values[index]

    return MetricSummary(
        label="B-factor" if use_bfactor else "pLDDT",
        minimum=min(values),
        mean=sum(values) / len(values),
        maximum=max(values),
        low_anchor=percentile(0.05) if use_bfactor else None,
        mid_anchor=percentile(0.50) if use_bfactor else None,
        high_anchor=percentile(0.95) if use_bfactor else None,
    )


def xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def line_points(start: int, end: int, x0: float, cell: float) -> tuple[float, float]:
    x = x0 + (start - 1) * cell
    w = max(cell, (end - start + 1) * cell)
    return x, w


# SVG primitives for the secondary-structure track.
def draw_helix(x: float, y: float, width: float, height: float, fill: str) -> str:
    r = min(height / 2.0, width / 3.0)
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{r:.2f}" ry="{r:.2f}" fill="{fill}" stroke="none" />'
    )


def draw_strand(x: float, y: float, width: float, height: float, fill: str) -> str:
    head = min(width * 0.35, max(height, 10.0))
    body = max(0.0, width - head)
    points = [
        (x, y + height * 0.2),
        (x + body, y + height * 0.2),
        (x + body, y),
        (x + width, y + height / 2.0),
        (x + body, y + height),
        (x + body, y + height * 0.8),
        (x, y + height * 0.8),
    ]
    pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
    return f'<polygon points="{pts}" fill="{fill}" stroke="none" />'


def draw_ss_fill_rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    # Slight overlap suppresses anti-aliased seams between adjacent confidence bands.
    return (
        f'<rect x="{x - 0.35:.2f}" y="{y:.2f}" width="{width + 0.7:.2f}" '
        f'height="{height:.2f}" fill="{fill}" stroke="none" />'
    )


def helix_clip_path(clip_id: str, x: float, y: float, width: float, height: float) -> str:
    r = min(height / 2.0, width / 3.0)
    return (
        f'<clipPath id="{clip_id}">'
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'rx="{r:.2f}" ry="{r:.2f}" />'
        f"</clipPath>"
    )


def strand_clip_path(clip_id: str, x: float, y: float, width: float, height: float) -> str:
    head = min(width * 0.35, max(height, 10.0))
    body = max(0.0, width - head)
    points = [
        (x, y + height * 0.2),
        (x + body, y + height * 0.2),
        (x + body, y),
        (x + width, y + height / 2.0),
        (x + body, y + height),
        (x + body, y + height * 0.8),
        (x, y + height * 0.8),
    ]
    pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
    return f'<clipPath id="{clip_id}"><polygon points="{pts}" /></clipPath>'


def draw_coil(x: float, y: float, width: float, height: float, fill: str) -> str:
    mid = y + height / 2.0
    return (
        f'<line x1="{x:.2f}" y1="{mid:.2f}" x2="{x + width:.2f}" y2="{mid:.2f}" '
        f'stroke="{fill}" stroke-width="2.4" stroke-linecap="round" />'
    )


def draw_missing(x: float, y: float, width: float, height: float, fill: str) -> str:
    mid = y + height / 2.0
    return (
        f'<line x1="{x:.2f}" y1="{mid:.2f}" x2="{x + width:.2f}" y2="{mid:.2f}" '
        f'stroke="{fill}" stroke-width="2.2" stroke-dasharray="4 3" stroke-linecap="round" />'
    )


def draw_clipped_metric_run(
    parts: List[str],
    clip_path: str,
    clip_id: str,
    chunk: Sequence[ResidueInfo],
    start_idx: int,
    end_idx: int,
    y: float,
    height: float,
    x0: float,
    cell: float,
    code: str,
    use_metric: bool,
    use_bfactor: bool,
    metric_summary: Optional[MetricSummary],
) -> None:
    parts.append(clip_path)
    parts.append(f'<g clip-path="url(#{clip_id})">')
    for idx in range(start_idx, end_idx + 1):
        color = metric_color(
            chunk[idx].plddt if use_metric else None, code, use_bfactor, metric_summary
        )
        rx, rw = line_points(idx + 1, idx + 1, x0, cell)
        parts.append(draw_ss_fill_rect(rx, y, rw, height, color))
    parts.append("</g>")


# Render the final vector figure row-by-row.
def render_svg(
    chain_id: str,
    residues: List[ResidueInfo],
    source_name: str,
    wrap: int,
    use_metric: bool,
    use_bfactor: bool,
    metric_summary: Optional[MetricSummary],
) -> str:
    wrap = max(20, wrap)
    x0 = 70.0
    cell = 10.8
    track_h = 13.0
    header_h = 78.0
    row_h = 78.0
    legend_h = 72.0 if use_metric else 34.0
    width = x0 + min(len(residues), wrap) * cell + 40.0
    rows = math.ceil(len(residues) / wrap)
    height = header_h + rows * row_h + legend_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<style><![CDATA['
        '.title{font:600 18px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#111827;}'
        '.subtitle{font:12px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#4b5563;}'
        '.label{font:11px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#374151;}'
        '.seq{font:12px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#111827;}'
        '.seq-missing{font:12px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#9ca3af;}'
        '.num{font:10px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#6b7280;}'
        ']]></style>',
        f'<text x="{x0:.1f}" y="28" class="title">{xml(source_name)} chain {xml(chain_id)}</text>',
        f'<text x="{x0:.1f}" y="47" class="subtitle">Secondary structure above sequence. Dashed segments mark unmodeled residues.</text>',
    ]
    if metric_summary:
        parts.append(
            f'<text x="{x0:.1f}" y="64" class="subtitle">{metric_summary.label} colors: min {metric_summary.minimum:.1f}, mean {metric_summary.mean:.1f}, max {metric_summary.maximum:.1f}</text>'
        )

    for row in range(rows):
        start = row * wrap + 1
        end = min(len(residues), (row + 1) * wrap)
        chunk = residues[start - 1 : end]
        base_y = header_h + row * row_h
        ss_y = base_y
        seq_y = base_y + 31.0
        num_y = base_y + 55.0
        parts.append(
            f'<text x="18" y="{seq_y:.2f}" class="label">{start}-{end}</text>'
        )
        i = 0
        run_id = 0
        while i < len(chunk):
            current = chunk[i]
            code = current.ss
            j = i
            while j + 1 < len(chunk) and chunk[j + 1].ss == code:
                j += 1
            x, w = line_points(i + 1, j + 1, x0, cell)
            if code == "H":
                clip_id = f"clip_{sanitize(chain_id)}_{row}_{run_id}"
                draw_clipped_metric_run(
                    parts,
                    helix_clip_path(clip_id, x, ss_y + 2.0, w, track_h),
                    clip_id,
                    chunk,
                    i,
                    j,
                    ss_y + 2.0,
                    track_h,
                    x0,
                    cell,
                    code,
                    use_metric,
                    use_bfactor,
                    metric_summary,
                )
            elif code == "E":
                clip_id = f"clip_{sanitize(chain_id)}_{row}_{run_id}"
                draw_clipped_metric_run(
                    parts,
                    strand_clip_path(clip_id, x, ss_y + 2.0, w, track_h),
                    clip_id,
                    chunk,
                    i,
                    j,
                    ss_y + 2.0,
                    track_h,
                    x0,
                    cell,
                    code,
                    use_metric,
                    use_bfactor,
                    metric_summary,
                )
            elif code == "M":
                color = metric_color(
                    current.plddt if use_metric else None, code, use_bfactor, metric_summary
                )
                parts.append(draw_missing(x, ss_y + 2.0, w, track_h, color))
            else:
                color = metric_color(
                    current.plddt if use_metric else None, code, use_bfactor, metric_summary
                )
                parts.append(draw_coil(x, ss_y + 2.0, w, track_h, color))
            run_id += 1
            i = j + 1
        for offset, residue in enumerate(chunk):
            xpos = x0 + offset * cell + 0.9
            seq_class = "seq" if residue.modeled else "seq-missing"
            parts.append(
                f'<text x="{xpos:.2f}" y="{seq_y:.2f}" class="{seq_class}">{residue.aa}</text>'
            )
        for pos in range(start, end + 1):
            xpos = x0 + (pos - start) * cell + cell / 2.0
            if pos == start or pos % 10 == 0:
                parts.append(
                    f'<line x1="{xpos:.2f}" y1="{seq_y + 4:.2f}" x2="{xpos:.2f}" y2="{seq_y + 10:.2f}" '
                    f'stroke="#9ca3af" stroke-width="1" />'
                )
                parts.append(
                    f'<text x="{xpos - 6:.2f}" y="{num_y:.2f}" class="num">{pos}</text>'
                )
            elif pos % 5 == 0:
                parts.append(
                    f'<line x1="{xpos:.2f}" y1="{seq_y + 5:.2f}" x2="{xpos:.2f}" y2="{seq_y + 9:.2f}" '
                    f'stroke="#9ca3af" stroke-width="1" />'
                )

    legend_y = header_h + rows * row_h + 10.0
    neutral = "#8b949e"
    parts.append(draw_helix(x0, legend_y, 26.0, 12.0, neutral))
    parts.append(f'<text x="{x0 + 34:.2f}" y="{legend_y + 10:.2f}" class="label">helix</text>')
    parts.append(draw_strand(x0 + 100, legend_y, 26.0, 12.0, neutral))
    parts.append(f'<text x="{x0 + 134:.2f}" y="{legend_y + 10:.2f}" class="label">strand</text>')
    parts.append(draw_coil(x0 + 200, legend_y, 26.0, 12.0, neutral))
    parts.append(f'<text x="{x0 + 234:.2f}" y="{legend_y + 10:.2f}" class="label">coil</text>')
    parts.append(draw_missing(x0 + 290, legend_y, 26.0, 12.0, "#666666"))
    parts.append(f'<text x="{x0 + 324:.2f}" y="{legend_y + 10:.2f}" class="label">unmodeled</text>')

    if use_metric:
        metric_y = legend_y + 40.0
        parts.append(
            f'<text x="{x0 - 6:.2f}" y="{metric_y:.2f}" class="label">{xml(metric_summary.label if metric_summary else "Metric")}</text>'
        )
        key_x = x0 + 62.0
        if use_bfactor and metric_summary is not None:
            labels = [
                f"<{metric_summary.low_anchor:.1f}",
                f"{metric_summary.mid_anchor:.1f}",
                f">{metric_summary.high_anchor:.1f}",
            ]
            colors = ["#004cff", "#e5e7eb", "#dc0000"]
            for idx, (color, label) in enumerate(zip(colors, labels)):
                x = key_x + idx * 116.0
                parts.append(
                    f'<rect x="{x:.2f}" y="{metric_y - 10:.2f}" width="32" height="12" fill="{color}" stroke="#6b7280" stroke-width="0.8" />'
                )
                parts.append(
                    f'<text x="{x + 42:.2f}" y="{metric_y:.2f}" class="label">{xml(label)}</text>'
                )
        else:
            labels = ["<50", "50-70", "70-90", ">90"]
            for idx, ((_, _, color), label) in enumerate(zip(AF_COLORS, labels)):
                x = key_x + idx * 72.0
                parts.append(
                    f'<rect x="{x:.2f}" y="{metric_y - 10:.2f}" width="24" height="10" fill="{color}" stroke="none" />'
                )
                parts.append(
                    f'<text x="{x + 30:.2f}" y="{metric_y:.2f}" class="label">{xml(label)}</text>'
                )

    parts.append("</svg>")
    return "\n".join(parts)


# Output stage: prefer PDF, but preserve SVG when requested or needed as fallback.
def export_pdf(svg_path: Path, pdf_path: Path) -> bool:
    converter = shutil.which("rsvg-convert")
    if converter:
        try:
            subprocess.run(
                [converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        return True
    except Exception:
        return False


def process_chain(
    structure: gemmi.Structure,
    ss_overrides: Dict[str, Dict[int, str]],
    chain_id: str,
    input_path: Path,
    output_dir: Path,
    prefix: str,
    wrap: int,
    make_pdf: bool,
    keep_svg: bool,
    use_bfactor: bool,
) -> None:
    # Prepare per-chain sequence/annotation data, render SVG, then convert to PDF
    # when a converter is available.
    chain = find_chain(structure[0], chain_id)
    residues, has_metric = residue_records(structure, chain, use_bfactor)
    assign_secondary_structure(structure, chain, residues, ss_overrides.get(chain_id))
    metric_summary = summarize_metric(residues, use_bfactor) if has_metric else None
    stem = f"{prefix}_chain{sanitize(chain_id)}"
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    svg = render_svg(
        chain_id,
        residues,
        input_path.name,
        wrap,
        has_metric,
        use_bfactor,
        metric_summary,
    )
    svg_path.write_text(svg, encoding="utf-8")
    if make_pdf and export_pdf(svg_path, pdf_path):
        print(f"wrote {pdf_path}")
        if not keep_svg:
            svg_path.unlink()
        else:
            print(f"wrote {svg_path}")
    else:
        if make_pdf:
            print(
                "warning: PDF export unavailable; wrote SVG fallback instead",
                file=sys.stderr,
            )
        print(f"wrote {svg_path}")


def main() -> int:
    # End-to-end pipeline: load structure, derive/supplement secondary structure,
    # select chains, and write one output file per chain.
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structure, ss_overrides = with_secondary_structure(
        input_path, gemmi.read_structure(str(input_path))
    )
    chains = polymer_chains(structure[0])
    if not chains:
        raise SystemExit("no polymer chains found")
    selected = extract_chain_ids(args.chain) or [chain.name for chain in chains]
    prefix = args.prefix or input_path.stem
    for chain_id in selected:
        process_chain(
            structure=structure,
            ss_overrides=ss_overrides,
            chain_id=chain_id,
            input_path=input_path,
            output_dir=output_dir,
            prefix=prefix,
            wrap=args.wrap,
            make_pdf=True,
            keep_svg=args.svg,
            use_bfactor=args.bfac,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
