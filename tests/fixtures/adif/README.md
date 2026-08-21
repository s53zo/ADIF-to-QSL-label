# ADIF conformance fixtures

These small fixtures exercise ADI grammar rules from the official Amateur Data
Interchange Format specification, version 3.1.7.

- Source: https://www.adif.org/317/ADIF_317.htm
- Current-version redirect: https://adif.org/adif
- Retrieved: 2026-08-21

The files are original, minimal test data derived from the specification's ADI
field, header, record, data-length, type-indicator, and user-defined-field rules;
they are not copies of a third-party log. Tests run only against these checked-in
fixtures and never fetch specification data from the network.

- `conformance.adi`: header, mixed-case descriptors, optional type indicators,
  empty and international values, multiple records, an unknown application field,
  and tag-like text inside a length-delimited value.
- `malformed-length.adi`: a declared value length that exceeds the remaining
  input, used to verify bounded parsing and diagnostics.
