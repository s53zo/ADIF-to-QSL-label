# QSL Label Generator (Avery Zweckform 3664 but configurable)

Generate **print-ready QSL labels** from your ADIF log.

- 📄 Outputs a **PDF** formatted for **Avery Zweckform 3664** (A4, 3×8 grid, 70×33.8 mm labels).
- 📡 Groups QSOs by callsign (alphabetical).
- 📝 Up to **4 QSOs per label** with fields: **Date | Time | Band | Mode | QSL**.
- 🖨️ Supports **fine-tuning alignment** (per column & per row) to fix printer offsets.
- ✍️ Automatic text shrinking prevents overflow into neighboring columns.

---

## Features

- **Input**: ADIF log (`.adi`)
- **Output**: PDF file
- **Table format**:
  ```
  To Radio                  HISCALL

  Date   Time   Band   Mode   QSL
  2025-07-19 15:48 40M   SSB    PSE
  2025-07-20 07:13 20M   CW     TNX
  2025-07-21 18:02 20M   FT8    PSE
  2025-07-22 19:40 40M   SSB    TNX

  Comment text                   TNX & 73
  ```

---

## Installation

```bash
git clone https://github.com/yourusername/qsl-labels.git
cd qsl-labels
pip install reportlab
```

---

## Usage

### Basic run

```bash
python make_qsl_labels.py --adif "log.adi" --out "qsl_labels.pdf"
```

### Debug print

Generate outlines and row guides (for test alignment on plain paper):

```bash
python make_qsl_labels.py --adif "log.adi" --out "qsl_labels.pdf" --outline --guides
```

---

## Fine-Tuning Alignment

Every printer is slightly different. You can adjust **columns** and **rows** independently in **millimeters**.

- **Per-column offsets**: shift each of the 3 columns left/right  
- **Per-row offsets**: shift each of the 8 rows up/down  

### Example

```bash
# Move column 1 → 0.6 mm right
# Column 2 → unchanged
# Column 3 → 0.4 mm left
# Row 5 → 0.8 mm up
python make_qsl_labels.py   --adif "log.adi" --out "qsl_labels.pdf"   --col-offsets "0.6,0,-0.4"   --row-offsets "0,0,0,0,0.8,0,0,0"   --outline
```

### Adjustment tips

1. Print with `--outline --guides` on plain paper.  
2. Hold print against an actual Avery 3664 sheet in the light.  
3. Measure where columns/rows are off.  
4. Use `--col-offsets` and `--row-offsets` to nudge them.  
5. Repeat until alignment is perfect.  
6. Turn off debug flags and print on real labels.

---

## Options

| Option            | Description |
|-------------------|-------------|
| `--adif PATH`     | Input ADIF log file |
| `--out PATH`      | Output PDF file |
| `--outline`       | Draw label borders (debug) |
| `--guides`        | Draw row guides (debug) |
| `--col-offsets`   | Comma-separated offsets per column in mm (e.g. `"0.5,0,-0.3"`) |
| `--row-offsets`   | Comma-separated offsets per row in mm (8 values for Avery 3664) |

---

## License

MIT – feel free to fork and adapt.
