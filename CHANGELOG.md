# Changelog

## v5.25 — 2026-08-21

- Add schema-versioned snapshots with safe legacy migrations and clear
  compatibility diagnostics.
- Restore every persistent filter, layout, display, and card/label setting.
- Preserve configured custom ADIF column sources even when the active log does
  not contain those fields.
- Upgrade the pinned, self-hosted jsPDF runtime from 2.5.1 to 4.2.1.
- Replace full-page base64 PDF conversion with bounded Blob/typed-array page
  processing, progress feedback, duplicate-export protection, and cancellation.
- Make ADIF parsing length-aware and preserve untouched fields, type indicators,
  headers, and records during export.
- Remove ADIF upload analytics, use local runtime/assets, add input limits, and
  improve mobile layout, contrast, reduced motion, and labelling.
- Add unit, migration, conformance, seeded property, PDF, responsive Playwright,
  and axe tests with ESLint, Prettier, Dependabot, and CI.

## v5.24 — 2026-08-21

- Fix direct QSL card snapshots so their saved row and column offsets match the
  single-card layout.
- Restore compatibility with direct QSL card snapshots created by older
  versions that contain label-grid offset arrays.
