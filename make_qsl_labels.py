#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSL label PDF generator for Avery Zweckform 3664 (A4, 3×8 grid; 70×33.8 mm)

Highlights
----------
• ADIF → print-ready PDF (grouped by CALL, alphabetical)
• 4 QSOs per label by default (configurable), no RST unless you add it
• Dynamic column widths per label (measure headers+data), always left-aligned
• "Shrink-only" layout: leaves free space on the right if columns don't need it
• Avoids printer edge clipping with left/right page margins
• Global and per-column/per-row fine tuning offsets (mm)
• Clear, single-point configuration + optional YAML config override
• Debug helpers: label outlines, row guides, left-edge ticks

Install
-------
pip install reportlab pyyaml
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Optional YAML config support
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # graceful fallback if PyYAML isn't installed

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm

# ============================================================
# ================  CONFIG: single source  ===================
# ============================================================

CONFIG: Dict = {
    # ----- PAGE -----
    "page_size": "A4",              # "A4" or "LETTER"
    "cols": 3,
    "rows": 8,
    # Label height from Avery 3664; label width is derived from page width minus margins
    "label_h_mm": 33.8,

    # Non-printable margins + global nudges
    "left_margin_mm": 3.0,          # add margin so outer labels don't get clipped
    "right_margin_mm": 3.0,         # NEW: right margin
    "top_margin_mm": None,          # None => auto-center vertically
    "x_offset_mm": 0.0,             # global +X moves right, -X moves left
    "y_offset_mm": 5.0,             # global +Y moves up (default 5mm up)

    # ----- GRID FINE TUNING -----
    "col_offsets_mm": [0.0, 0.0, 0.0],     # per column shift (mm): +right / -left
    "row_offsets_mm": [0.0] * 8,           # per row shift (mm):    +up    / -down

    # ----- INSIDE LABEL -----
    "pad_x_mm": 2.0,
    "pad_y_mm": 2.0,

    # Footer placement & texts
    "footer_left":  "SY: Blondie, South Adriatic sea",
    "footer_right": "TNX & 73",
    "footer_right_shift_mm": 5.0,   # move right footer LEFT by this many mm
    "footer_y_shift_mm": 2.0,       # raise both footers by this many mm (mm)

    # ----- TABLE (fully configurable columns) -----
    "rows_per_label": 4,

    # Columns: each entry is {"header": "...", "source": "..."}
    # "source" can be one of computed keys: DATE, TIME, BAND, MODE, QSL
    # or any raw ADIF field name such as RST_SENT, RST_RCVD, OPERATOR, etc.
    "columns": [
        {"header": "Date", "source": "DATE"},
        {"header": "Time", "source": "TIME"},
        {"header": "Band", "source": "BAND"},
        {"header": "QSL",  "source": "QSL"},   # place QSL next to Band
        {"header": "Mode", "source": "MODE"},
        # To include RST, uncomment (and update min/static widths accordingly):
        # {"header": "RSTs", "source": "RST_SENT"},
        # {"header": "RSTr", "source": "RST_RCVD"},
    ],

    # Column sizing
    "dynamic_col_widths": True,            # measure per label
    "min_col_mm":  [12, 10, 10, 6, 10],    # minimum width per column (mm) — must match len(columns)
    "static_col_mm": [20, 12, 12, 8, 18],  # used if dynamic_col_widths=False; must match len(columns)
    "col_slack_pt": 2.0,                   # extra breathing space per column when measuring (points)
    "shrink_only": True,                   # only scale down when too wide; otherwise leave free space

    # ----- TYPOGRAPHY -----
    "font_body": "Helvetica",
    "font_bold": "Helvetica-Bold",
    "font_mono": "Courier",
    "size_to_radio": 7.5,
    "size_callsign": 14,
    "size_col_headers": 7.2,
    "size_rows": 8.2,
    "size_footer": 7.2,

    # ----- LOGIC -----
    "qsl_received_keys": ["QSL_RCVD", "LOTW_QSL_RCVD", "EQSL_QSL_RCVD"],
    "ssb_modes": ["USB", "LSB"],

    # ----- DEBUG -----
    "draw_label_outlines": False,
    "draw_table_guides": False,
    "draw_cell_left_ticks": False,
}

# ============================================================
# ===================  ADIF utilities  =======================
# ============================================================

import sys

FIELD_TAG = re.compile(r"<([A-Za-z0-9_]+):(\d+)(?::[^>]+)?>", re.IGNORECASE)

def parse_adif(text: str) -> List[Dict[str, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    records_raw = re.split(r"(?i)<eor>", text)
    records: List[Dict[str, str]] = []
    for rec in records_raw:
        pos = 0
        fields: Dict[str, str] = {}
        while True:
            m = FIELD_TAG.search(rec, pos)
            if not m:
                break
            tag = m.group(1).upper()
            ln = int(m.group(2))
            start = m.end()
            val = rec[start:start+ln]
            pos = start + ln
            fields[tag] = val.strip()
        if fields:
            records.append(fields)
    return records

def fmt_date(adif_date: str) -> str:
    if not adif_date:
        return ""
    s = adif_date.strip()
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 and s[:8].isdigit() else s

def fmt_time(adif_time: str) -> str:
    if not adif_time:
        return ""
    s = adif_time.strip()
    return f"{s[0:2]}:{s[2:4]}" if len(s) >= 4 and s[:4].isdigit() else s

def norm_mode(mode: str, submode: str) -> str:
    mode = (mode or "").upper().strip()
    submode = (submode or "").upper().strip()
    if submode:
        return submode
    if mode in CONFIG["ssb_modes"]:
        return "SSB"
    return mode

def band_from_freq(freq_str: str) -> str:
    if not freq_str:
        return ""
    try:
        f = float(freq_str)
    except ValueError:
        return ""
    bands = [
        (1.8,   2.0,   "160M"),
        (3.5,   4.0,   "80M"),
        (5.3,   5.406, "60M"),
        (7.0,   7.3,   "40M"),
        (10.1,  10.15, "30M"),
        (14.0,  14.35, "20M"),
        (18.068,18.168,"17M"),
        (21.0,  21.45, "15M"),
        (24.89, 24.99, "12M"),
        (28.0,  29.7,  "10M"),
        (50.0,  54.0,  "6M"),
        (70.0,  71.0,  "4M"),
        (144.0, 148.0, "2M"),
        (420.0, 450.0, "70CM"),
    ]
    for lo, hi, b in bands:
        if lo <= f <= hi:
            return b
    return ""

def qsl_value(rec: Dict[str, str], received_keys: List[str]) -> str:
    for k in received_keys:
        if rec.get(k, "").upper() == "Y":
            return "TNX"
    return "PSE"

# ============================================================
# ================  Column / width helpers  ==================
# ============================================================

def get_cell_text(row: Dict[str, str], source: str, cfg: Dict) -> str:
    """Resolve display text for a column.
       Supports computed keys: DATE, TIME, BAND, MODE, QSL
       Or any raw ADIF field name (e.g., RST_SENT, RST_RCVD, OPERATOR)."""
    key = (source or "").upper().strip()

    if key == "DATE": return row.get("DATE", "")
    if key == "TIME": return row.get("TIME", "")
    if key == "BAND": return row.get("BAND", "")
    if key == "MODE": return row.get("MODE", "")
    if key == "QSL":  return row.get("QSL", "")

    # Raw ADIF passthrough
    return row.get(key, "")

from reportlab.pdfgen import canvas

def compute_col_widths_for_label(
    c: canvas.Canvas,
    cfg: Dict,
    qso_rows: List[Dict[str, str]],
    label_w_pt: float
) -> List[float]:
    """Compute per-label column widths (points).
       - Measure headers + data
       - Add slack
       - Enforce per-column minimums
       - If too wide: scale DOWN proportionally
       - Else: leave free space on the right (do NOT stretch)"""
    cols = cfg["columns"]
    n = len(cols)

    # 1) header widths (bold)
    widths: List[float] = []
    for col in cols:
        h = col["header"]
        w_header = c.stringWidth(h, cfg["font_bold"], cfg["size_col_headers"])
        widths.append(w_header)

    # 2) data widths (mono)
    for row in qso_rows or []:
        if not row: continue
        for i, col in enumerate(cols):
            txt = get_cell_text(row, col["source"], cfg) or ""
            w = c.stringWidth(txt, cfg["font_mono"], cfg["size_rows"])
            if w > widths[i]:
                widths[i] = w

    # 3) slack
    slack = float(cfg.get("col_slack_pt", 2.0))
    widths = [w + slack for w in widths]

    # 4) minimums
    min_mm = cfg.get("min_col_mm", [0] * n)
    if len(min_mm) != n:
        raise ValueError(f"min_col_mm must have {n} entries to match columns")
    min_pts = [m * mm for m in min_mm]
    for i in range(n):
        widths[i] = max(widths[i], min_pts[i])

    # 5) shrink if needed
    total = sum(widths)
    if total <= 0:
        static_mm = cfg.get("static_col_mm", [0] * n)
        if len(static_mm) != n:
            raise ValueError(f"static_col_mm must have {n} entries to match columns")
        return [w * mm for w in static_mm]

    shrink_only = bool(cfg.get("shrink_only", True))
    if (not shrink_only) or (total > label_w_pt):
        scale = min(1.0, label_w_pt / total) if shrink_only else (label_w_pt / total)
        widths = [w * scale for w in widths]

    return widths

# ============================================================
# ===================  PDF rendering  ========================
# ============================================================

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm

def page_size_tuple(name: str):
    name = (name or "A4").upper()
    if name == "A4": return A4
    if name in ("LETTER", "USLETTER", "US_LETTER"): return letter
    raise ValueError(f"Unsupported page size: {name}")

def build_labels(rows: List[Dict[str, str]], rows_per_label: int) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Group rows by CALL (alphabetical) and split into chunks of rows_per_label."""
    by_call = defaultdict(list)
    for r in rows:
        by_call[r["CALL"]].append(r)
    for c in by_call:
        by_call[c].sort(key=lambda x: (x["_DATE_RAW"], x["_TIME_RAW"]))
    labels: List[Tuple[str, List[Dict[str, str]]]] = []
    for call in sorted(by_call.keys()):
        items = by_call[call]
        for i in range(0, len(items), rows_per_label):
            labels.append((call, items[i:i+rows_per_label]))
    return labels

def shrink_and_draw(c, text: str, font: str, size: float,
                    x: float, y: float, max_width: float,
                    *, min_size: float = 5.0, step: float = 0.3) -> float:
    """Left-align draw: shrink text until it fits max_width, then draw at (x, y)."""
    if not text:
        return size
    txt = str(text)
    current = float(size)
    c.setFont(font, current)
    while c.stringWidth(txt, font, current) > max_width and current > min_size:
        current -= step
        if current < min_size:
            current = min_size
        c.setFont(font, current)
    c.drawString(x, y, txt)
    return current

def render_pdf(labels, cfg: Dict, out_pdf: Path):
    PAGE_W, PAGE_H = page_size_tuple(cfg["page_size"])
    page_width_mm = PAGE_W / mm

    # Compute label width from page width minus left/right margins
    usable_w_mm = page_width_mm - (cfg["left_margin_mm"] + cfg["right_margin_mm"])
    if usable_w_mm <= 0:
        raise ValueError("Left + right margins exceed page width.")
    LABEL_W = (usable_w_mm / cfg["cols"]) * mm
    LABEL_H = cfg["label_h_mm"] * mm

    COLS = cfg["cols"]
    ROWS = cfg["rows"]

    # Vertical margin / centering
    total_labels_h = ROWS * LABEL_H
    if cfg["top_margin_mm"] is None:
        vertical_free = (PAGE_H - total_labels_h)
        top_margin = vertical_free / 2.0
    else:
        top_margin = cfg["top_margin_mm"] * mm

    left_margin_pt = cfg["left_margin_mm"] * mm
    pad_x = cfg["pad_x_mm"] * mm
    pad_y = cfg["pad_y_mm"] * mm

    if len(cfg["col_offsets_mm"]) != COLS:
        raise ValueError(f"col_offsets_mm must have length {COLS}")
    if len(cfg["row_offsets_mm"]) != ROWS:
        raise ValueError(f"row_offsets_mm must have length {ROWS}")

    c = canvas.Canvas(str(out_pdf), pagesize=(PAGE_W, PAGE_H))
    labels_per_page = COLS * ROWS

    def draw_one(col_idx: int, row_idx: int, hiscall: str, qso_rows: List[Dict[str, str]]):
        # Base origin for this label cell
        x0 = left_margin_pt + col_idx * LABEL_W
        y0 = PAGE_H - top_margin - (row_idx + 1) * LABEL_H

        # Apply global offsets then per-column/row fine-tuning
        x0 += cfg["x_offset_mm"] * mm
        y0 += cfg["y_offset_mm"] * mm
        x0 += cfg["col_offsets_mm"][col_idx] * mm
        y0 += cfg["row_offsets_mm"][row_idx] * mm

        # Per-label column widths
        if cfg["dynamic_col_widths"]:
            col_w = compute_col_widths_for_label(c, cfg, qso_rows, LABEL_W)
        else:
            mm_list = cfg["static_col_mm"]
            if len(mm_list) != len(cfg["columns"]):
                raise ValueError("static_col_mm length must match columns length")
            scale = (LABEL_W / (sum(mm_list) * mm)) if sum(mm_list) > 0 else 1.0
            col_w = [w * mm * scale for w in mm_list]

        # Debug outline
        if cfg["draw_label_outlines"]:
            c.setLineWidth(0.3)
            c.setStrokeGray(0.85)
            c.rect(x0, y0, LABEL_W, LABEL_H, stroke=1, fill=0)
            c.setStrokeGray(0.0)

        x = x0 + pad_x
        y = y0 + LABEL_H - pad_y

        # Header
        c.setFont(cfg["font_body"], cfg["size_to_radio"])
        c.drawString(x, y - 8, "To Radio")
        c.setFont(cfg["font_bold"], cfg["size_callsign"])
        c.drawCentredString(x0 + LABEL_W/2, y - 8, hiscall.upper())

        # Table headers (LEFT-aligned)
        c.setFont(cfg["font_bold"], cfg["size_col_headers"])
        cx = x0
        for i, col in enumerate(cfg["columns"]):
            c.drawString(cx + pad_x, y - 20, col["header"])
            cx += col_w[i]

        # Optional left-edge ticks
        if cfg["draw_cell_left_ticks"]:
            cx = x0
            c.setLineWidth(0.3)
            c.setStrokeGray(0.2)
            for idx_col, _ in enumerate(cfg["columns"]):
                tick_x = cx + pad_x
                c.line(tick_x, y0 + pad_y, tick_x, y0 + LABEL_H - pad_y)
                cx += col_w[idx_col]
            c.setStrokeGray(0.0)

        # Data rows (LEFT-aligned)
        line_gap = 9
        first_line_y = y - 32
        row_y = [first_line_y - i*line_gap for i in range(cfg["rows_per_label"])]

        for ridx in range(cfg["rows_per_label"]):
            data = qso_rows[ridx] if ridx < len(qso_rows) else None
            cx = x0
            for i, col in enumerate(cfg["columns"]):
                cell_txt = get_cell_text(data or {}, col["source"], cfg) if data else ""
                max_w = col_w[i] - 2*pad_x
                if max_w < 1: max_w = col_w[i] - 1
                shrink_and_draw(
                    c, cell_txt, cfg["font_mono"], cfg["size_rows"],
                    cx + pad_x, row_y[ridx], max_w
                )
                cx += col_w[i]

        # Footer
        c.setFont(cfg["font_body"], cfg["size_footer"])
        foot_y = y0 + pad_y + 2 + (cfg.get("footer_y_shift_mm", 0.0) * mm)
        c.drawString(x, foot_y, cfg["footer_left"])
        c.drawRightString(x0 + LABEL_W - pad_x - (cfg.get("footer_right_shift_mm", 0.0) * mm),
                          foot_y, cfg["footer_right"])

    for idx, (call, chunk) in enumerate(labels):
        pos = idx % labels_per_page
        col = pos % COLS
        row = pos // COLS
        draw_one(col, row, call, chunk)
        if pos == labels_per_page - 1:
            c.showPage()
    c.showPage()
    c.save()

# ============================================================
# ====================  Main workflow  =======================
# ============================================================

def merge_dict(dst: Dict, src: Dict):
    """Shallow merge dicts (for simple, flat config fields)."""
    for k, v in src.items():
        dst[k] = v

def parse_float_list_csv(s: str) -> List[float]:
    if s is None or not s.strip():
        return []
    return [float(x.strip()) for x in s.split(",")]

def load_yaml_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load --config. Install with: pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping at top level")
    return data

def main():
    ap = argparse.ArgumentParser(description="Generate QSL label PDF from ADIF (Avery 3664)")
    ap.add_argument("--adif", required=True, help="Input ADIF file (.adi)")
    ap.add_argument("--out", required=True, help="Output PDF file")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config file")

    # Debug visuals
    ap.add_argument("--outline", action="store_true", help="Draw label outlines (debug)")
    ap.add_argument("--guides", action="store_true", help="Draw row guides (debug)")
    ap.add_argument("--left-ticks", action="store_true", help="Draw left-edge ticks (debug)")

    # Fine tuning overrides
    ap.add_argument("--col-offsets", type=str, default=None, help="Comma-separated mm offsets per column")
    ap.add_argument("--row-offsets", type=str, default=None, help="Comma-separated mm offsets per row")
    ap.add_argument("--x-offset-mm", type=float, default=None, help="Global horizontal shift (+ right, - left)")
    ap.add_argument("--y-offset-mm", type=float, default=None, help="Global vertical shift (+ up, - down)")
    ap.add_argument("--left-margin-mm", type=float, default=None, help="Override left page margin (mm)")
    ap.add_argument("--right-margin-mm", type=float, default=None, help="Override right page margin (mm)")

    # Dynamic columns toggle
    ap.add_argument("--static-cols", action="store_true", help="Disable dynamic widths (use static_col_mm)")

    args = ap.parse_args()

    # Load YAML config if provided (flat overrides only)
    cfg = dict(CONFIG)  # copy defaults
    if args.config:
        ycfg = load_yaml_config(Path(args.config))
        merge_dict(cfg, ycfg)

    # Apply CLI toggles
    if args.outline: cfg["draw_label_outlines"] = True
    if args.guides: cfg["draw_table_guides"] = True
    if args.left_ticks: cfg["draw_cell_left_ticks"] = True

    if args.col_offsets:
        offs = parse_float_list_csv(args.col_offsets)
        if len(offs) != cfg["cols"]:
            raise ValueError(f"--col-offsets must have exactly {cfg['cols']} values")
        cfg["col_offsets_mm"] = offs
    if args.row_offsets:
        offs = parse_float_list_csv(args.row_offsets)
        if len(offs) != cfg["rows"]:
            raise ValueError(f"--row-offsets must have exactly {cfg['rows']} values")
        cfg["row_offsets_mm"] = offs

    if args.x_offset_mm is not None: cfg["x_offset_mm"] = args.x_offset_mm
    if args.y_offset_mm is not None: cfg["y_offset_mm"] = args.y_offset_mm
    if args.left_margin_mm is not None: cfg["left_margin_mm"] = args.left_margin_mm
    if args.right_margin_mm is not None: cfg["right_margin_mm"] = args.right_margin_mm

    if args.static_cols: cfg["dynamic_col_widths"] = False

    # Load ADIF
    adif_path = Path(args.adif)
    out_pdf = Path(args.out)
    text = adif_path.read_text(encoding="utf-8", errors="ignore")
    recs = parse_adif(text)

    # Build rows
    rows: List[Dict[str, str]] = []
    for r in recs:
        call = r.get("CALL", "").upper().strip()
        if not call:
            continue
        date_raw = r.get("QSO_DATE", "") or r.get("QSO_DATE_OFF", "")
        time_raw = r.get("TIME_ON", "") or r.get("TIME_OFF", "")
        row = {
            # "computed" fields for columns
            "CALL": call,
            "_DATE_RAW": date_raw,
            "_TIME_RAW": time_raw,
            "DATE": fmt_date(date_raw),
            "TIME": fmt_time(time_raw),
            "BAND": (r.get("BAND","") or "").upper().strip() or band_from_freq(r.get("FREQ","")),
            "MODE": norm_mode(r.get("MODE",""), r.get("SUBMODE","")),
            "QSL":  qsl_value(r, cfg["qsl_received_keys"]),
        }
        # include all raw ADIF fields so they can be used as sources
        for k, v in r.items():
            if k not in row:
                row[k] = v
        rows.append(row)

    # Group & render
    labels = build_labels(rows, cfg["rows_per_label"])
    render_pdf(labels, cfg, out_pdf)
    print(f"Done. Wrote: {out_pdf.resolve()}")

if __name__ == "__main__":
    main()
