#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSL label PDF generator for Avery Zweckform 3664 (A4, 3×8 grid; 70×33.8 mm)

• Parses ADIF, groups QSOs by CALL (alphabetical).
• 4 QSOs per label (Date | Time | Band | Mode | QSL).
• Dynamic per-label column widths (based on content), constant within label.
• Text auto-shrinks to fit cell width.
• Fine-tuning: per-column (horizontal) and per-row (vertical) offsets.
• TABLE COLUMNS ARE ALWAYS LEFT-ALIGNED (headers + data).
• Debug: outline boxes, row guides, and optional left-edge ticks.

Install:  pip install reportlab
"""

from __future__ import annotations
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import mm

# ========================== CONFIG ==========================

CONFIG = {
    # Page/grid
    "page_size": "A4",
    "cols": 3,
    "rows": 8,
    "label_w_mm": 70.0,
    "label_h_mm": 33.8,
    "left_margin_mm": 0.0,
    "top_margin_mm": None,  # None => auto-center vertically

    # In-label padding
    "pad_x_mm": 2.0,
    "pad_y_mm": 2.0,

    # Debug visuals
    "draw_label_outlines": False,
    "draw_table_guides": False,
    "draw_cell_left_ticks": False,  # vertical ticks at each column's left edge

    # Table (no RST): Date | Time | Band | Mode | QSL
    "rows_per_label": 4,
    "table_headers": ["Date", "Time", "Band", "Mode", "QSL"],

    # Fallback static widths (mm) if dynamic disabled
    "table_col_mm": [20, 12, 12, 18, 8],

    # Dynamic per-label column sizing
    "dynamic_col_widths": True,
    "min_col_mm": [12, 10, 10, 12, 6],  # per-column minimums (mm)

    # Fonts/sizes
    "font_body": "Helvetica",
    "font_bold": "Helvetica-Bold",
    "font_mono": "Courier",
    "size_to_radio": 7.5,
    "size_callsign": 14,
    "size_col_headers": 7.2,
    "size_rows": 8.2,
    "size_footer": 7.2,

    # Static texts
    "label_left_header": "To Radio",
    "footer_left": "SY: Blondie, South Adriatic sea",
    "footer_right": "TNX & 73",

    # QSL logic
    "qsl_received_keys": ["QSL_RCVD", "LOTW_QSL_RCVD", "EQSL_QSL_RCVD"],

    # Mode normalization
    "ssb_modes": {"USB", "LSB"},

    # Fine-tuning (mm)
    "col_offsets_mm": [0.0, 0.0, 0.0],  # +right / -left per column
    "row_offsets_mm": [0.0] * 8,        # +up    / -down per row
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
    if not adif_date: return ""
    s = adif_date.strip()
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 and s[:8].isdigit() else s

def fmt_time(adif_time: str) -> str:
    if not adif_time: return ""
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

# =================== DRAWING UTILITIES ======================

def shrink_and_draw(c, text: str, font: str, size: float,
                    x: float, y: float, max_width: float,
                    *, min_size: float = 5.0, step: float = 0.3) -> float:
    """
    Left-align draw: shrink text until it fits max_width, then draw at (x, y).
    Returns the final font size used.
    """
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

def compute_col_widths_for_label(c: canvas.Canvas, cfg: Dict,
                                 qso_rows: List[Dict[str, str]], label_w_pt: float) -> List[float]:
    """
    Measure widest header/data text per column, add small slack, enforce minimums,
    then scale columns to fill label width exactly. Constant within the label.
    """
    headers = cfg["table_headers"]
    n = len(headers)
    widths = []
    for i in range(n):
        w_header = c.stringWidth(headers[i], cfg["font_bold"], cfg["size_col_headers"])
        widths.append(w_header)
    for row in qso_rows:
        if not row: continue
        parts = [row["DATE"], row["TIME"], row["BAND"], row["MODE"], row["QSL"]]
        for i, txt in enumerate(parts):
            w = c.stringWidth(txt or "", cfg["font_mono"], cfg["size_rows"])
            widths[i] = max(widths[i], w)
    widths = [w + 2.0 for w in widths]  # slack
    min_pts = [m * mm for m in cfg["min_col_mm"]]
    for i in range(n):
        widths[i] = max(widths[i], min_pts[i])
    total = sum(widths)
    if total <= 0:
        total_mm = sum(cfg["table_col_mm"]) or 1.0
        scale = (cfg["label_w_mm"] / total_mm)
        return [w * mm * scale for w in cfg["table_col_mm"]]
    scale = label_w_pt / total
    return [w * scale for w in widths]

# ====================== PDF RENDERING =======================

def page_size_tuple(name: str):
    if name.upper() == "A4":
        return A4
    if name.upper() in ("LETTER", "USLETTER", "US_LETTER"):
        return letter
    raise ValueError(f"Unsupported page size: {name}")

def build_labels(rows: List[Dict[str, str]], rows_per_label: int) -> List[Tuple[str, List[Dict[str, str]]]]:
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

def render_pdf(labels, cfg, out_pdf: Path):
    PAGE_W, PAGE_H = page_size_tuple(cfg["page_size"])
    LABEL_W = cfg["label_w_mm"] * mm
    LABEL_H = cfg["label_h_mm"] * mm
    COLS = cfg["cols"]
    ROWS = cfg["rows"]

    # margins
    if cfg["top_margin_mm"] is None:
        vertical_free = PAGE_H - ROWS * LABEL_H
        top_margin = vertical_free / 2.0
    else:
        top_margin = cfg["top_margin_mm"] * mm
    left_margin = cfg["left_margin_mm"] * mm

    pad_x = cfg["pad_x_mm"] * mm
    pad_y = cfg["pad_y_mm"] * mm

    if len(cfg["col_offsets_mm"]) != COLS:
        raise ValueError(f"col_offsets_mm must have length {COLS}")
    if len(cfg["row_offsets_mm"]) != ROWS:
        raise ValueError(f"row_offsets_mm must have length {ROWS}")

    c = canvas.Canvas(str(out_pdf), pagesize=(PAGE_W, PAGE_H))
    labels_per_page = COLS * ROWS

    def draw_one(col_idx: int, row_idx: int, hiscall: str, qso_rows: List[Dict[str, str]]):
        x0 = left_margin + col_idx * LABEL_W
        y0 = PAGE_H - top_margin - (row_idx + 1) * LABEL_H
        x0 += cfg["col_offsets_mm"][col_idx] * mm   # +right / -left
        y0 += cfg["row_offsets_mm"][row_idx] * mm   # +up    / -down

        # per-label column widths
        if cfg["dynamic_col_widths"]:
            col_w = compute_col_widths_for_label(c, cfg, qso_rows, LABEL_W)
        else:
            total_mm = sum(cfg["table_col_mm"]) or 1.0
            scale = (cfg["label_w_mm"] / total_mm)
            col_w = [w * mm * scale for w in cfg["table_col_mm"]]

        # debug outline
        if cfg["draw_label_outlines"]:
            c.setLineWidth(0.3)
            c.setStrokeGray(0.85)
            c.rect(x0, y0, LABEL_W, LABEL_H, stroke=1, fill=0)
            c.setStrokeGray(0.0)

        x = x0 + pad_x
        y = y0 + LABEL_H - pad_y

        # header
        c.setFont(cfg["font_body"], cfg["size_to_radio"])
        c.drawString(x, y - 8, cfg["label_left_header"])
        c.setFont(cfg["font_bold"], cfg["size_callsign"])
        c.drawCentredString(x0 + LABEL_W/2, y - 8, hiscall.upper())

        # table headers (LEFT-aligned)
        c.setFont(cfg["font_bold"], cfg["size_col_headers"])
        cx = x0
        for i, head in enumerate(cfg["table_headers"]):
            c.drawString(cx + pad_x, y - 20, head)
            cx += col_w[i]

        # optional left-edge ticks for visual confirmation
        if cfg["draw_cell_left_ticks"]:
            cx = x0
            c.setLineWidth(0.3)
            c.setStrokeGray(0.2)
            for i in range(len(cfg["table_headers"])):
                tick_x = cx + pad_x
                c.line(tick_x, y0 + pad_y, tick_x, y0 + LABEL_H - pad_y)
                cx += col_w[i]
            c.setStrokeGray(0.0)

        # data rows (LEFT-aligned)
        line_gap = 9
        first_line_y = y - 32
        row_y = [first_line_y - i*line_gap for i in range(cfg["rows_per_label"])]

        for ridx in range(cfg["rows_per_label"]):
            data = qso_rows[ridx] if ridx < len(qso_rows) else None
            parts = ["", "", "", "", ""]
            if data:
                parts = [data["DATE"], data["TIME"], data["BAND"], data["MODE"], data["QSL"]]
            cx = x0
            for i, txt in enumerate(parts):
                max_w = col_w[i] - 2*pad_x
                if max_w < 1:
                    max_w = col_w[i] - 1
                shrink_and_draw(
                    c, txt or "", cfg["font_mono"], cfg["size_rows"],
                    cx + pad_x, row_y[ridx], max_w
                )
                cx += col_w[i]

        # footer
        c.setFont(cfg["font_body"], cfg["size_footer"])
        c.drawString(x, y0 + pad_y + 2, cfg["footer_left"])
        c.drawRightString(x0 + LABEL_W - pad_y, y0 + pad_y + 2, cfg["footer_right"])

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
    if s is None or not s.strip():
        return []
    return [float(x.strip()) for x in s.split(",")]

def main():
    ap = argparse.ArgumentParser(description="Generate QSL label PDF from ADIF (Avery 3664)")
    ap.add_argument("--adif", required=True, help="Input ADIF file")
    ap.add_argument("--out", required=True, help="Output PDF file")

    # Debug visuals
    ap.add_argument("--outline", action="store_true", help="Draw label outlines")
    ap.add_argument("--guides", action="store_true", help="Draw row guides")
    ap.add_argument("--left-ticks", action="store_true", help="Draw thin ticks at each column's left edge")

    # Fine tuning
    ap.add_argument("--col-offsets", type=str, default=None, help="Comma-separated mm offsets per column")
    ap.add_argument("--row-offsets", type=str, default=None, help="Comma-separated mm offsets per row (8 values)")

    # Dynamic columns override (optional)
    ap.add_argument("--static-cols", action="store_true",
                    help="Use static normalized widths instead of dynamic per-label widths")

    args = ap.parse_args()

    # Apply flags
    if args.outline: CONFIG["draw_label_outlines"] = True
    if args.guides: CONFIG["draw_table_guides"] = True
    if args.left_ticks: CONFIG["draw_cell_left_ticks"] = True
    if args.col_offsets:
        offs = parse_float_list(args.col_offsets)
        if len(offs) != CONFIG["cols"]:
            raise ValueError(f"--col-offsets must have exactly {CONFIG['cols']} values")
        CONFIG["col_offsets_mm"] = offs
    if args.row_offsets:
        offs = parse_float_list(args.row_offsets)
        if len(offs) != CONFIG["rows"]:
            raise ValueError(f"--row-offsets must have exactly {CONFIG['rows']} values")
        CONFIG["row_offsets_mm"] = offs
    if args.static_cols:
        CONFIG["dynamic_col_widths"] = False

    # Load ADIF and build rows
    adif_path = Path(args.adif)
    out_pdf = Path(args.out)
    text = adif_path.read_text(encoding="utf-8", errors="ignore")
    recs = parse_adif(text)

    rows = []
    for r in recs:
        call = r.get("CALL", "").upper().strip()
        if not call: continue
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
            "QSL": qsl_value(r, CONFIG["qsl_received_keys"]),
        })

    labels = build_labels(rows, CONFIG["rows_per_label"])
    render_pdf(labels, CONFIG, out_pdf)
    print(f"Done. Wrote: {out_pdf.resolve()}")

if __name__ == "__main__":
    main()
