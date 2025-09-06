#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSL label PDF generator for Avery Zweckform 3664 (A4, 3??8 grid; 70??33.8 mm)

KEY FEATURES
------------
??? Parses ADIF and groups QSOs by CALL (alphabetical).
??? 4 QSOs per label (columns: Date | Time | Band | Mode | QSL).  (No RST.)
??? Column widths are auto-normalized to fill the 70 mm label width (you can use any numbers).
??? Automatic text shrink to prevent overflow into neighboring columns.
??? Fine-tuning alignment:
    - Per-column horizontal offsets (mm) -> fix slight left/right drift per column.
    - Per-row vertical offsets (mm)     -> fix slight up/down drift per row.
  This lets you print a test and dial in perfect alignment on your printer.

WORKFLOW TO DIAL IN ALIGNMENT
-----------------------------
1) Run with --outline and --guides and print on plain paper.
2) Hold the print over a real label sheet against light to see where it???s off.
3) Adjust:
    - --col-offsets "c1,c2,c3" (mm). Positive moves RIGHT; negative moves LEFT.
    - --row-offsets "r1,r2,...,r8" (mm). Positive moves UP; negative moves DOWN.
4) Reprint until every column & row sits perfectly on label cuts.
5) Turn off --outline/--guides for final prints.

REQUIREMENTS
------------
pip install reportlab
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# PDF / layout
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm

# ========================== CONFIG ==========================

CONFIG = {
    # --- Page & grid (Avery 3664) ---
    "page_size": "A4",          # "A4" or "LETTER"
    "cols": 3,                  # labels across
    "rows": 8,                  # labels down
    "label_w_mm": 70.0,         # 70.0 mm (Avery 3664 width)
    "label_h_mm": 33.8,         # 33.8 mm (Avery 3664 height)
    # Margins: 3??70mm = 210mm, so left margin is typically 0 on A4.
    # top_margin_mm=None auto-centers vertically (nice for printers that are true to size)
    "left_margin_mm": 0.0,
    "top_margin_mm": None,      # None => auto-center vertically

    # --- In-label padding & debugging ---
    "pad_x_mm": 2.0,            # inner side padding (mm)
    "pad_y_mm": 2.0,            # inner top/bottom padding (mm)
    "draw_label_outlines": False,   # visible label borders for test prints
    "draw_table_guides": False,     # faint row baselines for test prints

    # --- Table (no RST): Date | Time | Band | Mode | QSL ---
    "rows_per_label": 4,
    "table_headers": ["Date", "Time", "Band", "Mode", "QSL"],
    # You can put any numbers here; they will be scaled to fill 70 mm.
    "table_col_mm":  [20, 12, 12, 18, 8],

    # --- Fonts & sizes ---
    "font_body": "Helvetica",
    "font_bold": "Helvetica-Bold",
    "font_mono": "Courier",
    "size_to_radio": 7.5,
    "size_callsign": 14,
    "size_col_headers": 7.2,
    "size_rows": 8.2,
    "size_footer": 7.2,

    # --- Static texts ---
    "label_left_header": "To Radio",
    "footer_left":  "SY: Blondie, South Adriatic sea",
    "footer_right": "TNX & 73",

    # --- QSL column logic ---
    # If any of these ADIF flags == 'Y', show TNX; otherwise PSE.
    "qsl_received_keys": ["QSL_RCVD", "LOTW_QSL_RCVD", "EQSL_QSL_RCVD"],

    # --- Mode normalization ---
    "ssb_modes": {"USB", "LSB"},

    # --- FINE-TUNING OFFSETS (mm) ---
    # Per-column horizontal offsets (length must equal 'cols').
    # + value moves RIGHT, - value moves LEFT for that column only.
    "col_offsets_mm": [0.0, 0.0, 0.0],
    # Per-row vertical offsets (length must equal 'rows').
    # + value moves UP, - value moves DOWN for that row only.
    "row_offsets_mm": [0.0] * 8,
}

# ======================= ADIF HELPERS =======================

FIELD_TAG = re.compile(r"<([A-Za-z0-9_]+):(\d+)(?::[^>]+)?>", re.IGNORECASE)

def parse_adif(text: str) -> List[Dict[str, str]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    records_raw = re.split(r"(?i)<eor>", text)
    records = []
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

# ====================== PDF RENDERING =======================

def page_size_tuple(name: str):
    name = (name or "A4").upper()
    if name == "A4":
        return A4
    if name in ("LETTER", "USLETTER", "US_LETTER"):
        return letter
    raise ValueError(f"Unsupported page size: {name}")

def build_labels(rows: List[Dict[str, str]], rows_per_label: int) -> List[Tuple[str, List[Dict[str, str]]]]:
    """Group rows by CALL (alphabetical), split into chunks of rows_per_label."""
    by_call = defaultdict(list)
    for r in rows:
        by_call[r["CALL"]].append(r)
    # chronological within each callsign
    for c in by_call:
        by_call[c].sort(key=lambda x: (x["_DATE_RAW"], x["_TIME_RAW"]))
    labels: List[Tuple[str, List[Dict[str, str]]]] = []
    for call in sorted(by_call.keys()):
        items = by_call[call]
        for i in range(0, len(items), rows_per_label):
            labels.append((call, items[i:i+rows_per_label]))
    return labels

def shrink_and_draw(c: canvas.Canvas, text: str, font: str, size: float,
                    x: float, y: float, max_width: float, min_size: float = 5.0, step: float = 0.3):
    """Draw text with auto-shrink to fit max_width. Won't go below min_size."""
    c.setFont(font, size)
    if c.stringWidth(text, font, size) <= max_width:
        c.drawString(x, y, text)
        return
    while size > min_size and c.stringWidth(text, font, size) > max_width:
        size -= step
    c.setFont(font, size)
    c.drawString(x, y, text)

def render_pdf(labels, cfg, out_pdf: Path):
    PAGE_W, PAGE_H = page_size_tuple(cfg["page_size"])
    LABEL_W = cfg["label_w_mm"] * mm
    LABEL_H = cfg["label_h_mm"] * mm
    COLS = cfg["cols"]
    ROWS = cfg["rows"]

    # Margins
    if cfg["top_margin_mm"] is None:
        vertical_free = PAGE_H - ROWS * LABEL_H
        top_margin = vertical_free / 2.0
    else:
        top_margin = cfg["top_margin_mm"] * mm
    left_margin = cfg["left_margin_mm"] * mm

    pad_x = cfg["pad_x_mm"] * mm
    pad_y = cfg["pad_y_mm"] * mm

    # Normalize column widths to fill label width
    total_mm = sum(cfg["table_col_mm"])
    scale = (cfg["label_w_mm"] / total_mm) if total_mm > 0 else 1.0
    col_w = [w * mm * scale for w in cfg["table_col_mm"]]
    headers = cfg["table_headers"]

    # Validate fine-tuning vectors
    if len(cfg["col_offsets_mm"]) != COLS:
        raise ValueError(f"col_offsets_mm must have length {COLS}")
    if len(cfg["row_offsets_mm"]) != ROWS:
        raise ValueError(f"row_offsets_mm must have length {ROWS}")

    c = canvas.Canvas(str(out_pdf), pagesize=(PAGE_W, PAGE_H))
    labels_per_page = COLS * ROWS

    def draw_one(col_idx: int, row_idx: int, hiscall: str, qso_rows: List[Dict[str, str]]):
        # Base cell corner
        x0 = left_margin + col_idx * LABEL_W
        y0 = PAGE_H - top_margin - (row_idx + 1) * LABEL_H
        # Apply per-column/per-row fine offsets
        x0 += cfg["col_offsets_mm"][col_idx] * mm   # + right, - left
        y0 += cfg["row_offsets_mm"][row_idx] * mm   # + up, - down

        # Debug outline & guides
        if cfg["draw_label_outlines"]:
            c.setLineWidth(0.3)
            c.setStrokeGray(0.85)
            c.rect(x0, y0, LABEL_W, LABEL_H, stroke=1, fill=0)
            c.setStrokeGray(0.0)

        x = x0 + pad_x
        y = y0 + LABEL_H - pad_y

        # Header
        c.setFont(cfg["font_body"], cfg["size_to_radio"])
        c.drawString(x, y - 8, cfg["label_left_header"])
        c.setFont(cfg["font_bold"], cfg["size_callsign"])
        c.drawCentredString(x0 + LABEL_W/2, y - 8, hiscall.upper())

        # Table headers
        c.setFont(cfg["font_bold"], cfg["size_col_headers"])
        cx = x0
        for i, head in enumerate(headers):
            c.drawString(cx + pad_x, y - 20, head)
            cx += col_w[i]

        # QSO rows
        line_gap = 9                  # vertical spacing between rows (pt)
        first_line_y = y - 32
        row_y = [first_line_y - i*line_gap for i in range(cfg["rows_per_label"])]

        if cfg["draw_table_guides"]:
            c.setStrokeGray(0.9)
            for yy in row_y:
                c.line(x0 + pad_x, yy - 2, x0 + LABEL_W - pad_x, yy - 2)
            c.setStrokeGray(0.0)

        for ridx in range(cfg["rows_per_label"]):
            data = qso_rows[ridx] if ridx < len(qso_rows) else None
            parts = ["", "", "", "", ""]
            if data:
                parts = [data["DATE"], data["TIME"], data["BAND"], data["MODE"], data["QSL"]]
            cx = x0
            for i, txt in enumerate(parts):
                shrink_and_draw(
                    c, txt, cfg["font_mono"], cfg["size_rows"],
                    cx + pad_x, row_y[ridx], col_w[i] - 2
                )
                cx += col_w[i]

        # Footer
        c.setFont(cfg["font_body"], cfg["size_footer"])
        c.drawString(x, y0 + pad_y + 2, cfg["footer_left"])
        c.drawRightString(x0 + LABEL_W - pad_x, y0 + pad_y + 2, cfg["footer_right"])

    # Paginate
    for idx, (call, chunk) in enumerate(labels):
        pos = idx % labels_per_page
        col = pos % COLS
        row = pos // COLS
        draw_one(col, row, call, chunk)
        if pos == labels_per_page - 1:
            c.showPage()
    c.showPage()
    c.save()

# ========================= MAIN ============================

def parse_float_list(s: str) -> List[float]:
    """Parse 'a,b,c' into [a,b,c]. Empty or None -> []."""
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",")]

def main():
    ap = argparse.ArgumentParser(
        description="Generate QSL label PDF from ADIF (Avery 3664, 3x8; per-column/row fine-tuning)."
    )
    ap.add_argument("--adif", required=True, help="Path to ADIF file (.adi)")
    ap.add_argument("--out", required=True, help="Output PDF path")

    # Debug / alignment aids
    ap.add_argument("--outline", action="store_true", help="Draw faint label rectangles")
    ap.add_argument("--guides", action="store_true", help="Draw baseline guides for table rows")

    # Optional: override per-column and per-row offsets from CLI
    ap.add_argument("--col-offsets", type=str, default=None,
                    help="Comma-separated mm offsets per column (e.g. '0.5,0,-0.3'); +right, -left")
    ap.add_argument("--row-offsets", type=str, default=None,
                    help="Comma-separated mm offsets per row (8 values on Avery 3664); +up, -down")

    args = ap.parse_args()

    # Apply debug toggles
    if args.outline:
        CONFIG["draw_label_outlines"] = True
    if args.guides:
        CONFIG["draw_table_guides"] = True

    # Apply fine-tuning if provided
    if args.col_offsets is not None:
        offs = parse_float_list(args.col_offsets)
        if len(offs) != CONFIG["cols"]:
            raise ValueError(f"--col-offsets must have exactly {CONFIG['cols']} values")
        CONFIG["col_offsets_mm"] = offs
    if args.row_offsets is not None:
        offs = parse_float_list(args.row_offsets)
        if len(offs) != CONFIG["rows"]:
            raise ValueError(f"--row-offsets must have exactly {CONFIG['rows']} values")
        CONFIG["row_offsets_mm"] = offs

    # Load ADIF
    adif_path = Path(args.adif)
    out_pdf = Path(args.out)
    text = adif_path.read_text(encoding="utf-8", errors="ignore")
    recs = parse_adif(text)

    # Extract fields
    rows: List[Dict[str, str]] = []
    for r in recs:
        call = r.get("CALL", "").upper().strip()
        if not call:
            continue
        date_raw = r.get("QSO_DATE", "") or r.get("QSO_DATE_OFF", "")
        time_raw = r.get("TIME_ON", "") or r.get("TIME_OFF", "")
        rows.append({
            "CALL": call,
            "_DATE_RAW": date_raw,
            "_TIME_RAW": time_raw,
            "DATE": fmt_date(date_raw),
            "TIME": fmt_time(time_raw),
            "BAND": (r.get("BAND","") or "").upper().strip() or band_from_freq(r.get("FREQ","")),
            "MODE": norm_mode(r.get("MODE",""), r.get("SUBMODE","")),
            "QSL":  qsl_value(r, CONFIG["qsl_received_keys"]),
        })

    # Build labels and render
    labels = build_labels(rows, CONFIG["rows_per_label"])
    render_pdf(labels, CONFIG, out_pdf)

    print(f"Done. Wrote: {out_pdf.resolve()}")

if __name__ == "__main__":
    main()
