# ADIF to QSL Labels

Browser-based, print-ready QSL labels and direct QSL cards generated locally
from an ADIF log.

Open the [hosted helper](https://s53zo.github.io/ADIF-to-QSL-label/make_qsl_labels.html)
or serve/clone this repository and open `make_qsl_labels.html`. The browser tool
is the maintained version; `make_qsl_labels.py` is retained as an outdated legacy
implementation.

## Features

- Label sheets and direct QSL cards with exact millimetre dimensions.
- Configurable grids, margins, gaps, per-row/per-column calibration, typography,
  layout, filters, sorting, deduplication, and QSL status rules.
- Custom table columns sourced from standard or application-defined ADIF tags.
- Versioned JSON snapshots that restore label/card modes, filters, custom
  columns, and layout settings. Legacy unversioned snapshots are migrated.
- Offline-first PDF generation using a pinned local copy of jsPDF 4.2.1.
- Loss-preserving ADIF export that adds `QSL_SENT=Y` and `QSL_SENT_VIA=B` to the
  selected records while retaining untouched fields and records.

All ADIF/config processing happens in the browser. The application does not
upload logs or snapshots and contains no analytics event reporting.

ADIF export preserves record order and all untouched field names, values, type
indicators, and extension fields. Updated records receive canonical field-length
descriptors; line endings, outer record whitespace, and trailing header
whitespace may be normalized. Filtering can intentionally omit records when
“only filtered QSOs” is selected.

> In the printer dialog, choose **Actual size / 100%**. Disable “Fit to page.”

## Browser workflow

1. Open `make_qsl_labels.html` from the hosted site or a local static server.
2. Load an `.adi`, `.adif`, or `.txt` log in **Data**.
3. Choose **Label sheets** or **Direct QSL card** and adjust the layout.
4. Add custom columns by selecting any discovered ADIF tag. A tag restored from
   a snapshot remains available even when the current log does not contain it.
5. Use **Save snapshot** to keep all stable settings and **Load snapshot** to
   restore them later.
6. Export the PDF. Multi-page exports show progress and can be cancelled.

Snapshots written by v5.25 use schema version 1. See
[`docs/config-schema.md`](docs/config-schema.md) for compatibility and migration
details.

## Local development

Requirements:

- Node.js 24 or another current LTS release
- npm
- Chromium installed through Playwright
- Poppler (`pdftoppm`) for PDF pixel comparison

```bash
npm ci
npx playwright install chromium
python3 -m http.server 4173 --bind 127.0.0.1
```

Then open <http://127.0.0.1:4173/make_qsl_labels.html>.

## Verification

```bash
npm run lint
npm run format:check
npm run test:unit
npm run test:e2e
```

Or run the same aggregate used by CI:

```bash
npm run ci
```

For a local working-tree comparison against the latest commit:

```bash
npm run benchmark:pdf
```

Set `PDF_BASELINE_REF` to compare against another Git reference.

The suite includes configuration migrations, label/card snapshot round trips,
official-spec-derived ADIF fixtures, seeded parser properties, untouched ADIF
field/record preservation, PDF page-box and preview-to-PDF pixel comparisons,
responsive Playwright scenarios, and axe accessibility checks.

PDF regression tests use a 0.1 mm page-size tolerance. The visual comparison
allows up to 6% changed pixels because PDF rasterization re-antialiases the 4×
canvas image; larger changes fail and produce diagnostic images under the
Playwright test output directory.

## Dependencies

The runtime PDF library is pinned and self-hosted:

- jsPDF 4.2.1, MIT license
- Runtime asset: `vendor/jspdf.umd.min.js`
- Package source: <https://www.npmjs.com/package/jspdf/v/4.2.1>

Development and test dependencies are pinned in `package-lock.json`.

## Legacy Python version

The old ReportLab-based script remains available for existing command-line
workflows but is not feature-equivalent with the browser tool:

```bash
pip install reportlab pyyaml
python make_qsl_labels.py --adif log.adi --out qsl_labels.pdf
```

## License

MIT — free to use, fork, and adapt.
