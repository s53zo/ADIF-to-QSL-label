# Configuration snapshot schema

## Current schema: version 1

Snapshots saved by application v5.25 and later contain an integer
`schemaVersion` at the top level. They contain stable user settings only; ADIF
records, filenames, generated labels, and transient UI/export state are not
stored.

Version 1 includes output mode and dimensions, grid calibration, table columns,
typography and layout, filters and QSL rules, sorting and deduplication, theme,
rendering options, and debug settings.

ADIF source identifiers are normalized to uppercase and must match
`[A-Z][A-Z0-9_]{0,63}`. Valid application-defined tags are retained even if the
currently loaded ADIF does not contain them.

## Migration policy

Snapshots without `schemaVersion` are treated as legacy version 0 and pass
through a pure v0 → v1 migration before validation.

Legacy direct-card snapshots sometimes stored `cols=1` and `rows=1` while
retaining label-grid offset arrays with three and eight entries. Grid offsets do
not apply in card mode, so the migration deterministically canonicalizes both to
`[0]`. The UI reports that the snapshot was migrated.

Label snapshots retain strict array-length validation because silently dropping
label calibration values would change printed placement.

Snapshots using a newer schema than the application supports are rejected with
an explicit compatibility message. Malformed JSON, invalid field sources,
invalid numeric arrays, and excessive nesting receive separate diagnostics.

## Round-trip guarantee

For current-schema snapshots, save → load → save produces the same normalized
configuration. Migration and round-trip behavior is covered in unit and
Playwright tests for both label and direct-card modes.
