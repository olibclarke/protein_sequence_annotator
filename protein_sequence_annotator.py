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
US_LETTER_PORTRAIT = (612.0, 792.0)
LEFT_MARGIN = 18.0
RIGHT_MARGIN = 28.0
ROW_GAP = 4.0
CHAR_WIDTH = 7.4
CELL_WIDTH = 10.8


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


@dataclass
class StructureRun:
    code: str
    start: int
    end: int
    label: Optional[str] = None


@dataclass
class LabelPlacement:
    row_index: int
    visible_start: int
    visible_end: int


MIN_LABELED_SEGMENT = 6


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
    parser.add_argument(
        "--label",
        action="store_true",
        help="Label helices and strands as α-1/β-1, α-2/β-2, etc.",
    )
    parser.add_argument(
        "--paginate",
        action="store_true",
        help="Split long outputs into multiple US-letter-friendly landscape pages.",
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
    has_metric = False
    for residue in chain.get_polymer():
        pos = residue_position(residue, len(residues))
        if pos is None:
            continue
        item = residues[pos - 1]
        item.modeled = True
        item.aa = aa1(residue.name)
        item.plddt = residue_bfactor(residue) if use_bfactor else residue_plddt(residue)
        has_metric = has_metric or item.plddt is not None
    return residues, has_metric


def chain_position_maps(
    chain: gemmi.Chain, sequence_len: int
) -> tuple[Dict[int, int], Dict[int, int]]:
    # Build lookup tables from author numbering and label_seq numbering onto the
    # displayed sequence positions.
    auth_to_pos: Dict[int, int] = {}
    label_to_pos: Dict[int, int] = {}
    for residue in chain.get_polymer():
        pos = residue_position(residue, sequence_len)
        if pos is None:
            continue
        auth_to_pos[residue.seqid.num] = pos
        if residue.label_seq:
            label_to_pos[residue.label_seq] = pos
    return auth_to_pos, label_to_pos


def sequence_number_map(chain: gemmi.Chain, sequence_len: int) -> List[int]:
    # Build the displayed residue numbering for each sequence position. Layout is
    # still driven by sequence position, but labels/ticks should follow the
    # author residue numbering when it exists.
    anchors: List[tuple[int, int]] = []
    for residue in chain.get_polymer():
        pos = residue_position(residue, sequence_len)
        if pos is not None:
            anchors.append((pos, residue.seqid.num))
    if not anchors:
        return list(range(1, sequence_len + 1))

    numbers = [0] * sequence_len
    first_pos, first_num = anchors[0]
    for pos in range(1, first_pos + 1):
        numbers[pos - 1] = first_num - (first_pos - pos)

    for (prev_pos, prev_num), (next_pos, next_num) in zip(anchors, anchors[1:]):
        for pos in range(prev_pos, next_pos + 1):
            numbers[pos - 1] = prev_num + (pos - prev_pos)

    last_pos, last_num = anchors[-1]
    for pos in range(last_pos, sequence_len + 1):
        numbers[pos - 1] = last_num + (pos - last_pos)

    return numbers


def resolve_annotation_position(
    seq_num: int,
    auth_to_pos: Dict[int, int],
    label_to_pos: Dict[int, int],
    sequence_len: int,
) -> Optional[int]:
    pos = label_to_pos.get(seq_num, auth_to_pos.get(seq_num, seq_num))
    return pos if 1 <= pos <= sequence_len else None


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


def secondary_structure_runs(
    residues: Sequence[ResidueInfo], include_labels: bool
) -> List[StructureRun]:
    # Extract contiguous H/E/M/C blocks after all sequence/annotation mapping is done.
    runs: List[StructureRun] = []
    start = 0
    helix_count = 0
    strand_count = 0
    while start < len(residues):
        code = residues[start].ss
        end = start
        while end + 1 < len(residues) and residues[end + 1].ss == code:
            end += 1
        label = None
        if include_labels and code == "H":
            helix_count += 1
            label = f"α-{helix_count}"
        elif include_labels and code == "E":
            strand_count += 1
            label = f"β-{strand_count}"
        runs.append(StructureRun(code=code, start=start + 1, end=end + 1, label=label))
        start = end + 1
    return runs


def choose_label_placements(
    runs: Sequence[StructureRun], wrap: int, total_rows: int
) -> Dict[int, List[LabelPlacement]]:
    # For wrapped elements, label each substantial visible segment. Skip labels
    # on tiny carry-over fragments where the label would be clutter.
    placements: Dict[int, List[LabelPlacement]] = {}
    for run_idx, run in enumerate(runs):
        if run.label is None or run.code not in {"H", "E"}:
            continue
        segment_placements: List[LabelPlacement] = []
        for row_index in range(total_rows):
            row_start = row_index * wrap + 1
            row_end = min((row_index + 1) * wrap, run.end)
            if run.end < row_start or run.start > row_end:
                continue
            visible_start = max(run.start, row_start)
            visible_end = min(run.end, row_end)
            visible_width = visible_end - visible_start + 1
            if visible_width >= MIN_LABELED_SEGMENT:
                segment_placements.append(
                    LabelPlacement(
                    row_index=row_index,
                    visible_start=visible_start,
                    visible_end=visible_end,
                    )
                )
        if not segment_placements:
            # If every wrapped fragment is short, at least label the widest one.
            best: Optional[LabelPlacement] = None
            best_width = -1
            for row_index in range(total_rows):
                row_start = row_index * wrap + 1
                row_end = min((row_index + 1) * wrap, run.end)
                if run.end < row_start or run.start > row_end:
                    continue
                visible_start = max(run.start, row_start)
                visible_end = min(run.end, row_end)
                visible_width = visible_end - visible_start + 1
                if visible_width > best_width:
                    best_width = visible_width
                    best = LabelPlacement(
                        row_index=row_index,
                        visible_start=visible_start,
                        visible_end=visible_end,
                    )
            if best is not None:
                segment_placements = [best]
        if segment_placements:
            placements[run_idx] = segment_placements
    return placements


# Merge DSSP-derived states with any explicit file annotations.
def assign_secondary_structure(
    structure: gemmi.Structure,
    chain: gemmi.Chain,
    residues: List[ResidueInfo],
    dssp_assignments: Optional[Dict[int, str]] = None,
) -> None:
    chain_name = chain.name
    auth_to_pos, label_to_pos = chain_position_maps(chain, len(residues))
    if dssp_assignments:
        for seq_num, code in dssp_assignments.items():
            pos = resolve_annotation_position(
                seq_num, auth_to_pos, label_to_pos, len(residues)
            )
            if pos is not None:
                residues[pos - 1].ss = code
    for helix in structure.helices:
        if helix.start.chain_name != chain_name or helix.end.chain_name != chain_name:
            continue
        start = resolve_annotation_position(
            helix.start.res_id.seqid.num, auth_to_pos, label_to_pos, len(residues)
        )
        end = resolve_annotation_position(
            helix.end.res_id.seqid.num, auth_to_pos, label_to_pos, len(residues)
        )
        if start is not None and end is not None:
            apply_ranges(residues, start, end, "H")
    for sheet in structure.sheets:
        for strand in sheet.strands:
            if strand.start.chain_name != chain_name or strand.end.chain_name != chain_name:
                continue
            start = resolve_annotation_position(
                strand.start.res_id.seqid.num, auth_to_pos, label_to_pos, len(residues)
            )
            end = resolve_annotation_position(
                strand.end.res_id.seqid.num, auth_to_pos, label_to_pos, len(residues)
            )
            if start is not None and end is not None:
                apply_ranges(residues, start, end, "E")
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


def row_bounds(
    residue_numbers: Sequence[int], wrap: int, row_start: int, row_end: int
) -> tuple[List[int], List[int]]:
    starts: List[int] = []
    ends: List[int] = []
    for absolute_row in range(row_start, row_end):
        start = absolute_row * wrap + 1
        end = min(len(residue_numbers), (absolute_row + 1) * wrap)
        if start > len(residue_numbers):
            break
        starts.append(residue_numbers[start - 1])
        ends.append(residue_numbers[end - 1])
    return starts, ends


def gutter_layout(
    residue_numbers: Sequence[int], wrap: int, row_start: int, row_end: int, cell: float
) -> tuple[float, float, float]:
    starts, ends = row_bounds(residue_numbers, wrap, row_start, row_end)
    left_width = max((len(str(value)) for value in starts), default=1) * CHAR_WIDTH
    right_width = max((len(str(value)) for value in ends), default=1) * CHAR_WIDTH
    left_number_x = LEFT_MARGIN + left_width
    x0 = left_number_x + ROW_GAP
    max_cols = max(1, min(wrap, len(residue_numbers) - row_start * wrap))
    content_width = x0 + max_cols * cell + ROW_GAP + right_width
    return left_number_x, x0, content_width


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


def structure_label_x(code: str, x: float, width: float) -> float:
    # Center labels on the full visible secondary-structure element span.
    return x + width / 2.0


def structure_label_optical_offset(code: str, label: Optional[str]) -> float:
    if code != "E" or not label:
        return 0.0
    # Greek beta labels read right-heavy; use a smaller correction once the
    # numeric suffix grows to two digits.
    suffix = label.split("-", 1)[-1]
    digits = len(suffix)
    return max(0.0, 4.0 - 1.5 * max(0, digits - 1))


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


def draw_continuation_marker(x: float, y: float, height: float, side: str, width: float) -> str:
    # Mark wrapped helices/strands with a subtle in-track ellipsis. This reads
    # as "continues" without competing with the secondary-structure shape.
    cy = y + height / 2.0
    radius = 1.25
    spacing = 4.0
    dot_count = 1 if width < 18.0 else 3
    total_span = (dot_count - 1) * spacing
    start_x = x + 5.0 if side == "left" else x - 5.0 - total_span
    parts = []
    for idx in range(dot_count):
        cx = start_x + idx * spacing
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="#ffffff" stroke="#6b7280" stroke-width="0.45" />'
        )
    return "\n".join(parts)


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
    residue_numbers: Sequence[int],
    source_name: str,
    wrap: int,
    use_metric: bool,
    use_bfactor: bool,
    metric_summary: Optional[MetricSummary],
    show_labels: bool,
    page_row_start: int = 0,
    rows_per_page: Optional[int] = None,
    page_index: int = 1,
    page_count: int = 1,
    fixed_page_size: Optional[tuple[float, float]] = None,
) -> str:
    wrap = max(20, wrap)
    cell = CELL_WIDTH
    track_h = 13.0
    row_h = 78.0
    legend_h = 72.0 if use_metric else 34.0
    subtitle_lines = [
        "Secondary structure above sequence. Dashed segments mark unmodeled residues."
    ]
    if fixed_page_size is not None:
        subtitle_lines = [
            "Secondary structure above sequence.",
            "Dashed segments mark unmodeled residues.",
        ]
    header_h = 96.0 if fixed_page_size is not None and use_metric else 84.0 if fixed_page_size is not None else 78.0
    total_rows = math.ceil(len(residues) / wrap)
    row_start = page_row_start
    row_end = total_rows if rows_per_page is None else min(total_rows, row_start + rows_per_page)
    rows = max(0, row_end - row_start)
    left_number_x, x0, content_width = gutter_layout(
        residue_numbers, wrap, row_start, row_end, cell
    )
    content_height = header_h + rows * row_h + legend_h
    if fixed_page_size is not None:
        width, height = fixed_page_size
    else:
        width, height = content_width + RIGHT_MARGIN, content_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<style><![CDATA['
        '.title{font:600 18px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#111827;}'
        '.subtitle{font:12px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#4b5563;}'
        '.label{font:11px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#374151;}'
        '.sslabel{font:10px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#4b5563;}'
        '.seq{font:12px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#111827;}'
        '.seq-missing{font:12px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#9ca3af;}'
        '.num{font:10px "Courier New","Liberation Mono",Menlo,Monaco,monospace;fill:#6b7280;}'
        ']]></style>',
        f'<text x="{x0:.1f}" y="28" class="title">{xml(source_name)} chain {xml(chain_id)}</text>',
    ]
    for i, line in enumerate(subtitle_lines):
        parts.append(
            f'<text x="{x0:.1f}" y="{47 + i * 14:.1f}" class="subtitle">{xml(line)}</text>'
        )
    if metric_summary:
        parts.append(
            f'<text x="{x0:.1f}" y="{47 + len(subtitle_lines) * 14 + 3:.1f}" class="subtitle">{metric_summary.label} colors: min {metric_summary.minimum:.1f}, mean {metric_summary.mean:.1f}, max {metric_summary.maximum:.1f}</text>'
        )
    if page_count > 1:
        parts.append(
            f'<text x="{width - RIGHT_MARGIN:.1f}" y="28" text-anchor="end" class="subtitle">page {page_index}/{page_count}</text>'
        )
    runs = secondary_structure_runs(residues, show_labels)
    label_placements = (
        choose_label_placements(runs, wrap, total_rows) if show_labels else {}
    )

    for local_row, absolute_row in enumerate(range(row_start, row_end)):
        start = absolute_row * wrap + 1
        end = min(len(residues), (absolute_row + 1) * wrap)
        chunk = residues[start - 1 : end]
        base_y = header_h + local_row * row_h
        ss_y = base_y
        seq_y = base_y + 31.0
        num_y = base_y + 50.0
        left_num = residue_numbers[start - 1]
        right_num = residue_numbers[end - 1]
        seq_right = x0 + (end - start + 1) * cell
        parts.append(
            f'<text x="{left_number_x:.2f}" y="{seq_y:.2f}" text-anchor="end" class="label">{left_num}-</text>'
        )
        parts.append(
            f'<text x="{seq_right + ROW_GAP:.2f}" y="{seq_y:.2f}" text-anchor="start" class="label">-{right_num}</text>'
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
            global_start = start + i
            global_end = start + j
            continued_left = global_start > 1 and residues[global_start - 2].ss == code
            continued_right = global_end < len(residues) and residues[global_end].ss == code
            if code == "H":
                clip_id = f"clip_{sanitize(chain_id)}_{absolute_row}_{run_id}"
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
                if continued_left:
                    parts.append(
                        draw_continuation_marker(x, ss_y + 2.0, track_h, "left", w)
                    )
                if continued_right:
                    parts.append(
                        draw_continuation_marker(
                            x + w, ss_y + 2.0, track_h, "right", w
                        )
                    )
            elif code == "E":
                clip_id = f"clip_{sanitize(chain_id)}_{absolute_row}_{run_id}"
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
                if continued_left:
                    parts.append(
                        draw_continuation_marker(x, ss_y + 2.0, track_h, "left", w)
                    )
                if continued_right:
                    parts.append(
                        draw_continuation_marker(
                            x + w, ss_y + 2.0, track_h, "right", w
                        )
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
        if show_labels:
            label_y = ss_y + 0.5
            for run_idx, run in enumerate(runs):
                if run.label is None or run.code not in {"H", "E"}:
                    continue
                placements = label_placements.get(run_idx, [])
                for placement in placements:
                    if placement.row_index != absolute_row:
                        continue
                    visible_start = placement.visible_start
                    visible_end = placement.visible_end
                    label_x, label_w = line_points(
                        visible_start - start + 1, visible_end - start + 1, x0, cell
                    )
                    xpos = structure_label_x(run.code, label_x, label_w)
                    xpos -= structure_label_optical_offset(run.code, run.label)
                    parts.append(
                        f'<text x="{xpos:.2f}" y="{label_y:.2f}" text-anchor="middle" class="sslabel">{xml(run.label)}</text>'
                    )
        for offset, residue in enumerate(chunk):
            xpos = x0 + offset * cell + cell / 2.0
            seq_class = "seq" if residue.modeled else "seq-missing"
            parts.append(
                f'<text x="{xpos:.2f}" y="{seq_y:.2f}" text-anchor="middle" class="{seq_class}">{residue.aa}</text>'
            )
        for pos in range(start, end + 1):
            xpos = x0 + (pos - start) * cell + cell / 2.0
            residue_num = residue_numbers[pos - 1]
            is_major = residue_num % 10 == 0
            is_minor = residue_num % 5 == 0 and not is_major
            if is_major:
                parts.append(
                    f'<line x1="{xpos:.2f}" y1="{seq_y + 4:.2f}" x2="{xpos:.2f}" y2="{seq_y + 10:.2f}" '
                    f'stroke="#9ca3af" stroke-width="1" />'
                )
                parts.append(
                    f'<text x="{xpos:.2f}" y="{num_y:.2f}" text-anchor="middle" class="num">{residue_num}</text>'
                )
            elif is_minor:
                parts.append(
                    f'<line x1="{xpos:.2f}" y1="{seq_y + 4:.2f}" x2="{xpos:.2f}" y2="{seq_y + 8:.2f}" '
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


def combine_pdfs(pdf_paths: Sequence[Path], output_path: Path) -> bool:
    if not pdf_paths:
        return False
    join = Path("/System/Library/Automator/Combine PDF Pages.action/Contents/MacOS/join")
    if not join.exists():
        return False
    try:
        subprocess.run(
            [str(join), "--output", str(output_path), *[str(path) for path in pdf_paths]],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
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
    show_labels: bool,
    paginate: bool,
) -> None:
    # Prepare per-chain sequence/annotation data, render SVG, then convert to PDF
    # when a converter is available.
    chain = find_chain(structure[0], chain_id)
    residues, has_metric = residue_records(structure, chain, use_bfactor)
    residue_numbers = sequence_number_map(chain, len(residues))
    assign_secondary_structure(structure, chain, residues, ss_overrides.get(chain_id))
    metric_summary = summarize_metric(residues, use_bfactor) if has_metric else None
    stem = f"{prefix}_chain{sanitize(chain_id)}"
    total_rows = math.ceil(len(residues) / max(20, wrap))
    effective_wrap = max(20, wrap)
    if paginate:
        page_width, page_height = US_LETTER_PORTRAIT
        max_num_width = max((len(str(value)) for value in residue_numbers), default=1) * CHAR_WIDTH
        gutter_width = LEFT_MARGIN + max_num_width + ROW_GAP + ROW_GAP + max_num_width
        effective_wrap = max(20, int((page_width - gutter_width - RIGHT_MARGIN) // CELL_WIDTH))
        total_rows = math.ceil(len(residues) / effective_wrap)
        page_header_h = 96.0 if has_metric else 84.0
        page_legend_h = 72.0 if has_metric else 34.0
        rows_per_page = max(1, int((page_height - page_header_h - page_legend_h - 12.0) // 78.0))
    else:
        rows_per_page = total_rows
    page_count = max(1, math.ceil(total_rows / rows_per_page))
    if page_count == 1:
        page_stem = stem
        svg_path = output_dir / f"{page_stem}.svg"
        pdf_path = output_dir / f"{page_stem}.pdf"
        svg = render_svg(
            chain_id,
            residues,
            residue_numbers,
            input_path.name,
            effective_wrap,
            has_metric,
            use_bfactor,
            metric_summary,
            show_labels,
            page_row_start=0,
            rows_per_page=rows_per_page,
            page_index=1,
            page_count=1,
            fixed_page_size=US_LETTER_PORTRAIT if paginate else None,
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
        return

    merged_pdf_path = output_dir / f"{stem}.pdf"
    page_pdf_paths: List[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for page_idx in range(page_count):
            page_name = f"{stem}_p{page_idx + 1:02d}"
            svg = render_svg(
                chain_id,
                residues,
                residue_numbers,
                input_path.name,
                effective_wrap,
                has_metric,
                use_bfactor,
                metric_summary,
                show_labels,
                page_row_start=page_idx * rows_per_page,
                rows_per_page=rows_per_page,
                page_index=page_idx + 1,
                page_count=page_count,
                fixed_page_size=US_LETTER_PORTRAIT,
            )
            if keep_svg:
                svg_path = output_dir / f"{page_name}.svg"
                svg_path.write_text(svg, encoding="utf-8")
                print(f"wrote {svg_path}")
                pdf_path = tmpdir_path / f"{page_name}.pdf"
            else:
                svg_path = tmpdir_path / f"{page_name}.svg"
                pdf_path = tmpdir_path / f"{page_name}.pdf"
                svg_path.write_text(svg, encoding="utf-8")
            if not make_pdf or not export_pdf(svg_path, pdf_path):
                fallback_svg = output_dir / f"{page_name}.svg"
                if not keep_svg:
                    fallback_svg.write_text(svg, encoding="utf-8")
                print(
                    "warning: PDF export unavailable; wrote paginated SVG fallback instead",
                    file=sys.stderr,
                )
                print(f"wrote {fallback_svg}")
                return
            page_pdf_paths.append(pdf_path)

        if combine_pdfs(page_pdf_paths, merged_pdf_path):
            print(f"wrote {merged_pdf_path}")
        else:
            for page_idx, page_pdf_path in enumerate(page_pdf_paths, start=1):
                fallback_pdf = output_dir / f"{stem}_p{page_idx:02d}.pdf"
                fallback_pdf.write_bytes(page_pdf_path.read_bytes())
                print(f"wrote {fallback_pdf}")
            print(
                "warning: could not combine paginated PDFs; kept separate page PDFs",
                file=sys.stderr,
            )


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
    available_chain_ids = [chain.name for chain in chains]
    selected = extract_chain_ids(args.chain) or [chain.name for chain in chains]
    missing = [chain_id for chain_id in selected if chain_id not in available_chain_ids]
    if missing:
        raise SystemExit(
            f"requested chain(s) not found: {', '.join(missing)}. "
            f"Available polymer chains: {', '.join(available_chain_ids)}"
        )
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
            show_labels=args.label,
            paginate=args.paginate,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
