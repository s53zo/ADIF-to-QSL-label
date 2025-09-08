# QSL Label Generator — Avery Zweckform 3664

Generate **print-ready QSL labels** from your ADIF log.

- 📄 Outputs a **PDF** aligned for **Avery Zweckform 3664** (A4, 3×8, 70×33.8 mm).
- 🧭 **Single-point config**: page → grid → offsets → label → table → typography → logic → debug.
- 🧮 **Dynamic columns** (per label), always **left-aligned**, **shrink-only** (free space on the right).
- 🎛️ Fine-tune **left/right page margins**, **global XY offsets**, **per-column** and **per-row** nudges.
- 🧱 Fully configurable **columns**: include RST or any ADIF field by editing `columns`.
- 🧪 Debug: **outlines**, **row guides**, **left-edge ticks** for quick calibration on plain paper.

> **Tip:** In the printer dialog select **Actual size / 100%** (no page scaling).

---

## Installation

```bash
pip install reportlab pyyaml
```

---

## Quick Start

```bash
python make_qsl_labels.py --adif "log.adi" --out "qsl_labels.pdf"
```

**What you get** by default:

- 3 columns × 8 rows of labels on A4
- Left/right margins = **3 mm** (avoids printer clipping)
- Global vertical offset = **+5 mm**
- 4 QSOs per label
- Columns: **Date | Time | Band | QSL | Mode**
- Dynamic per-label widths; free space on the right if unused

---

## Configuration (single place)

Open `make_qsl_labels.py` and edit the `CONFIG` dict at the top. Settings are grouped in the order you’ll calibrate:

1. **PAGE**: paper size, label grid (3×8), margins, global offsets  
2. **GRID FINE TUNING**: per-column/per-row shifts (mm)  
3. **INSIDE LABEL**: padding (mm)  
4. **TABLE**: rows per label, **columns** (headers + sources), sizing behavior  
5. **TYPOGRAPHY**: fonts & sizes  
6. **LOGIC**: QSL rules, SSB normalization  
7. **DEBUG**: outline/guides/ticks for test prints  

You can also supply a YAML file and override any fields:

```bash
python make_qsl_labels.py --config config.yaml --adif "log.adi" --out "qsl_labels.pdf"
```

### Adding RST (or any ADIF field)

Edit `CONFIG["columns"]` and add new entries:

```python
"columns": [
  {"header": "Date", "source": "DATE"},
  {"header": "Time", "source": "TIME"},
  {"header": "Band", "source": "BAND"},
  {"header": "RSTs", "source": "RST_SENT"},
  {"header": "RSTr", "source": "RST_RCVD"},
  {"header": "QSL",  "source": "QSL"},
  {"header": "Mode", "source": "MODE"},
]
"min_col_mm":  [12, 10, 10, 6, 6, 6, 10]
"static_col_mm":[20, 12, 12, 8, 8, 8, 18]
```

No other code changes are required.

---

## Useful CLI Flags

```bash
# Debug aids
--outline            # draw label outlines
--guides             # draw row baselines
--left-ticks         # draw left edge tick inside each column (visual left alignment)

# Page margins and global offsets
--left-margin-mm 3   # override left page margin
--right-margin-mm 3  # override right page margin
--x-offset-mm -1.5   # global shift left/right
--y-offset-mm  6     # global shift up/down

# Fine-tuning per column/row
--col-offsets "0,0,0"                # mm per column (3 values)
--row-offsets "0,0,0,0,0,0,0,0"      # mm per row    (8 values)

# Columns sizing
--static-cols        # disable dynamic widths; use CONFIG.static_col_mm
```

Example:

```bash
python make_qsl_labels.py \
  --adif "log.adi" \
  --out "labels.pdf" \
  --left-margin-mm 3 --right-margin-mm 3 \
  --x-offset-mm -1.5 --y-offset-mm 6 \
  --outline --left-ticks
```

---

## YAML Config (optional)

Create `config.yaml` to override any fields in `CONFIG`:

```yaml
# config.yaml
left_margin_mm: 3.0
right_margin_mm: 3.0
x_offset_mm: -1.0
y_offset_mm: 6.0

rows_per_label: 4
columns:
  - { header: "Date", source: "DATE" }
  - { header: "Time", source: "TIME" }
  - { header: "Band", source: "BAND" }
  - { header: "RSTs", source: "RST_SENT" }
  - { header: "RSTr", source: "RST_RCVD" }
  - { header: "QSL",  source: "QSL" }
  - { header: "Mode", source: "MODE" }
min_col_mm:  [12, 10, 10, 6, 6, 6, 10]
static_col_mm:[20, 12, 12, 8, 8, 8, 18]
```

Run with:

```bash
python make_qsl_labels.py --config config.yaml --adif "log.adi" --out "labels.pdf"
```

---

## Sample

A tiny `sample.adif` is included for quick testing:

```bash
python make_qsl_labels.py --adif sample.adif --out sample_labels.pdf --outline
```

---

## License

MIT – free to use, fork, and adapt.
