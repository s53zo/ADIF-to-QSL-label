(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.QSLCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const CONFIG_SCHEMA_VERSION = 1;
  const DEFAULT_COLUMN_SOURCES = [
    "DATE",
    "TIME",
    "BAND",
    "MODE",
    "QSL",
    "CALL",
    "FREQ",
    "RST_SENT",
    "RST_RCVD",
  ];
  const SOURCE_RE = /^[A-Z][A-Z0-9_]{0,63}$/;
  const BLOCKED_KEYS = new Set(["__proto__", "prototype", "constructor"]);
  const MAX_CONFIG_DEPTH = 24;
  const MAX_ADIF_BYTES = 50 * 1024 * 1024;
  const MAX_ADIF_RECORDS = 250000;

  class ConfigError extends Error {
    constructor(code, message) {
      super(message);
      this.name = "ConfigError";
      this.code = code;
    }
  }

  class AdifError extends Error {
    constructor(code, message) {
      super(message);
      this.name = "AdifError";
      this.code = code;
    }
  }

  function sanitizeJson(value, depth = 0) {
    if (depth > MAX_CONFIG_DEPTH) {
      throw new ConfigError("too-deep", "Config nesting is too deep.");
    }
    if (value === null || typeof value === "string" || typeof value === "boolean") return value;
    if (typeof value === "number") {
      if (!Number.isFinite(value))
        throw new ConfigError("invalid-number", "Config contains a non-finite number.");
      return value;
    }
    if (Array.isArray(value)) return value.map((item) => sanitizeJson(item, depth + 1));
    if (typeof value !== "object") {
      throw new ConfigError("invalid-value", "Config contains an unsupported value.");
    }
    const out = Object.create(null);
    for (const [key, item] of Object.entries(value)) {
      if (BLOCKED_KEYS.has(key)) continue;
      out[key] = sanitizeJson(item, depth + 1);
    }
    return out;
  }

  function normalizeMode(mode) {
    return String(mode || "labels").toLowerCase() === "qslcard" ? "qslCard" : "labels";
  }

  function normalizeSource(source) {
    return String(source == null ? "" : source)
      .trim()
      .toUpperCase();
  }

  function isValidSource(source) {
    return SOURCE_RE.test(normalizeSource(source));
  }

  function mergeColumnSources(savedSources = [], discoveredSources = [], defaults = DEFAULT_COLUMN_SOURCES) {
    const seen = new Set();
    const result = [];
    for (const raw of [...defaults, ...savedSources, ...discoveredSources]) {
      const source = normalizeSource(raw);
      if (!source || seen.has(source) || !isValidSource(source)) continue;
      seen.add(source);
      result.push(source);
    }
    return result;
  }

  function toNumberArray(value, key) {
    if (!Array.isArray(value)) throw new ConfigError("invalid-array", `${key} must be an array of numbers.`);
    return value.map((item) => {
      const number = Number(item);
      if (!Number.isFinite(number))
        throw new ConfigError("invalid-array", `${key} must be an array of numbers.`);
      return number;
    });
  }

  function migrateV0ToV1(legacy) {
    const value = sanitizeJson(legacy);
    const warnings = [];
    value.mode = normalizeMode(value.mode);
    if (value.mode === "qslCard") {
      const offsetsChanged =
        value.cols !== 1 ||
        value.rows !== 1 ||
        !Array.isArray(value.colOffsets) ||
        value.colOffsets.length !== 1 ||
        !Array.isArray(value.rowOffsets) ||
        value.rowOffsets.length !== 1;
      value.cols = 1;
      value.rows = 1;
      value.colOffsets = [0];
      value.rowOffsets = [0];
      if (offsetsChanged) warnings.push("Legacy card-only grid offsets were normalized.");
    }
    if (!value.filters && value.f && typeof value.f === "object" && !Array.isArray(value.f)) {
      value.filters = sanitizeJson(value.f);
      warnings.push("Legacy filter settings were migrated.");
    }
    delete value.f;
    value.schemaVersion = 1;
    return { value, warnings };
  }

  function validateConfig(config) {
    if (!config || typeof config !== "object" || Array.isArray(config)) {
      throw new ConfigError("not-object", "Config must be a JSON object.");
    }
    if (config.schemaVersion !== CONFIG_SCHEMA_VERSION) {
      throw new ConfigError(
        "unsupported-version",
        `Unsupported config schema version ${String(config.schemaVersion)}.`,
      );
    }
    config.mode = normalizeMode(config.mode);
    for (const key of ["cols", "rows", "rowsPerLabel"]) {
      if (config[key] !== undefined && (!Number.isInteger(config[key]) || config[key] <= 0)) {
        throw new ConfigError("invalid-value", `${key} must be a positive integer.`);
      }
    }
    if (config.columns !== undefined) {
      if (!Array.isArray(config.columns) || config.columns.length === 0) {
        throw new ConfigError("invalid-columns", "columns must be a non-empty array.");
      }
      config.columns = config.columns.map((column) => {
        if (!column || typeof column !== "object" || Array.isArray(column)) {
          throw new ConfigError("invalid-columns", "Each column must be an object.");
        }
        const header = String(column.header == null ? "" : column.header).trim();
        const source = normalizeSource(column.source);
        if (!header) throw new ConfigError("invalid-columns", "Each column needs a header.");
        if (!isValidSource(source)) {
          throw new ConfigError("invalid-source", `Invalid ADIF column source: ${source || "(empty)"}.`);
        }
        return { ...column, header: header.slice(0, 64), source };
      });
    }
    for (const key of ["colOffsets", "rowOffsets", "minColMM", "staticColMM"]) {
      if (config[key] !== undefined) config[key] = toNumberArray(config[key], key);
    }
    if (config.mode === "qslCard") {
      config.cols = 1;
      config.rows = 1;
    }
    if (config.columns) {
      if (config.minColMM && config.minColMM.length !== config.columns.length) {
        throw new ConfigError("length-mismatch", "minColMM length must match columns.");
      }
      if (config.staticColMM && config.staticColMM.length !== config.columns.length) {
        throw new ConfigError("length-mismatch", "staticColMM length must match columns.");
      }
    }
    if (config.cols && config.colOffsets && config.colOffsets.length !== config.cols) {
      throw new ConfigError("length-mismatch", "colOffsets length must match cols.");
    }
    if (config.rows && config.rowOffsets && config.rowOffsets.length !== config.rows) {
      throw new ConfigError("length-mismatch", "rowOffsets length must match rows.");
    }
    if (config.qsoRowsBold !== undefined) {
      const value = Number(config.qsoRowsBold);
      if (!Number.isInteger(value) || value < 0 || value > 2) {
        throw new ConfigError("invalid-value", "qsoRowsBold must be 0, 1, or 2.");
      }
      config.qsoRowsBold = value;
    }
    return config;
  }

  function migrateAndValidateConfig(input) {
    if (!input || typeof input !== "object" || Array.isArray(input)) {
      throw new ConfigError("not-object", "Config must be a JSON object.");
    }
    const rawVersion = input.schemaVersion;
    const version = rawVersion === undefined ? 0 : rawVersion;
    if (!Number.isInteger(version) || version < 0) {
      throw new ConfigError("invalid-version", "schemaVersion must be a non-negative integer.");
    }
    if (version > CONFIG_SCHEMA_VERSION) {
      throw new ConfigError(
        "future-version",
        `This snapshot uses schema version ${version}; this app supports up to ${CONFIG_SCHEMA_VERSION}.`,
      );
    }
    let value = sanitizeJson(input);
    const warnings = [];
    let current = version;
    while (current < CONFIG_SCHEMA_VERSION) {
      if (current === 0) {
        const migrated = migrateV0ToV1(value);
        value = migrated.value;
        warnings.push(...migrated.warnings);
        current = 1;
      } else {
        throw new ConfigError(
          "unsupported-version",
          `No migration is available from schema version ${current}.`,
        );
      }
    }
    return { value: validateConfig(value), warnings, migrated: version !== CONFIG_SCHEMA_VERSION };
  }

  function createSnapshotConfig(runtimeConfig, filters) {
    const clean = sanitizeJson(runtimeConfig);
    delete clean.f;
    clean.schemaVersion = CONFIG_SCHEMA_VERSION;
    clean.filters = sanitizeJson(filters || {});
    return validateConfig(clean);
  }

  function advanceCharacters(text, start, count) {
    let index = start;
    let remaining = count;
    while (remaining > 0 && index < text.length) {
      const code = text.codePointAt(index);
      index += code > 0xffff ? 2 : 1;
      remaining -= 1;
    }
    return { end: index, complete: remaining === 0 };
  }

  function parseDescriptor(text, index) {
    if (text[index] !== "<") return null;
    const close = text.indexOf(">", index + 1);
    if (close < 0) return null;
    const inside = text.slice(index + 1, close).trim();
    if (/^(EOH|EOR)$/i.test(inside)) {
      return { kind: inside.toUpperCase(), start: index, end: close + 1 };
    }
    const match = inside.match(/^([A-Za-z0-9_]+):(\d+)(?::([^>]*))?$/);
    if (!match) return null;
    return {
      kind: "FIELD",
      start: index,
      end: close + 1,
      tagOriginal: match[1],
      tag: match[1].toUpperCase(),
      length: Number(match[2]),
      attr: match[3] || "",
    };
  }

  function scanAdif(text, options = {}) {
    const source = String(text || "").replace(/\r\n?/g, "\n");
    const maxBytes = options.maxBytes || MAX_ADIF_BYTES;
    if (source.length > maxBytes)
      throw new AdifError("too-large", `ADIF input exceeds ${maxBytes} characters.`);
    const tokens = [];
    const warnings = [];
    let index = 0;
    while (index < source.length) {
      const start = source.indexOf("<", index);
      if (start < 0) break;
      const descriptor = parseDescriptor(source, start);
      if (!descriptor) {
        index = start + 1;
        continue;
      }
      if (descriptor.kind === "FIELD") {
        const valueRange = advanceCharacters(source, descriptor.end, descriptor.length);
        descriptor.valueStart = descriptor.end;
        descriptor.valueEnd = valueRange.end;
        descriptor.value = source.slice(descriptor.valueStart, descriptor.valueEnd);
        descriptor.rawEnd = descriptor.valueEnd;
        if (!valueRange.complete) {
          warnings.push(`Field ${descriptor.tag} declares more characters than remain in the file.`);
        }
        tokens.push(descriptor);
        index = descriptor.valueEnd;
      } else {
        descriptor.rawEnd = descriptor.end;
        tokens.push(descriptor);
        index = descriptor.end;
      }
    }
    return { source, tokens, warnings };
  }

  function parseRecordFields(recordText) {
    const { source, tokens, warnings } = scanAdif(recordText, { maxBytes: MAX_ADIF_BYTES });
    const fields = tokens
      .filter((token) => token.kind === "FIELD")
      .map((token) => ({
        tag: token.tag,
        tagOriginal: token.tagOriginal,
        attr: token.attr,
        value: token.value,
      }));
    const last = tokens.length ? tokens[tokens.length - 1].rawEnd : 0;
    return { fields, trailing: source.slice(last), warnings };
  }

  function parseAdifDocument(text, options = {}) {
    const scan = scanAdif(text, options);
    const { source, tokens, warnings } = scan;
    const eoh = tokens.find((token) => token.kind === "EOH");
    const bodyStart = eoh ? eoh.end : 0;
    const records = [];
    let recordStart = bodyStart;
    for (const token of tokens) {
      if (token.start < bodyStart || token.kind !== "EOR") continue;
      const raw = source.slice(recordStart, token.start);
      const parsed = parseRecordFields(raw);
      if (parsed.fields.length || raw.trim()) {
        records.push({ raw, fields: parsed.fields, trailing: parsed.trailing });
        if (records.length > (options.maxRecords || MAX_ADIF_RECORDS)) {
          throw new AdifError("too-many-records", "ADIF input contains too many records.");
        }
      }
      recordStart = token.end;
    }
    const remainder = source.slice(recordStart);
    if (remainder.trim()) {
      const parsed = parseRecordFields(remainder);
      if (parsed.fields.length)
        records.push({ raw: remainder, fields: parsed.fields, trailing: parsed.trailing });
    }
    return {
      source,
      hasHeader: Boolean(eoh),
      headerRaw: eoh ? source.slice(0, eoh.start) : "",
      records,
      warnings,
    };
  }

  function recordToObject(record) {
    const result = Object.create(null);
    for (const field of record.fields || []) result[field.tag] = String(field.value || "").trim();
    return result;
  }

  function parseADIF(text, options) {
    return parseAdifDocument(text, options).records.map(recordToObject);
  }

  function formatAdifField(tag, value, attr) {
    const cleanTag = normalizeSource(tag);
    const val = value == null ? "" : String(value);
    const type = attr ? `:${String(attr)}` : "";
    return `<${cleanTag}:${Array.from(val).length}${type}>${val}`;
  }

  function ensureQslFields(recordText) {
    const parsed = parseRecordFields(recordText);
    const fields = parsed.fields.map((field) => ({ ...field }));
    const upsert = (tag, value) => {
      const existing = fields.find((field) => field.tag === tag);
      if (existing) existing.value = value;
      else fields.push({ tag, tagOriginal: tag, attr: "", value });
    };
    upsert("QSL_SENT", "Y");
    upsert("QSL_SENT_VIA", "B");
    return (
      fields
        .map((field) => formatAdifField(field.tagOriginal || field.tag, field.value, field.attr))
        .join("") + (parsed.trailing || "")
    );
  }

  function buildUpdatedAdif(options) {
    const records = Array.isArray(options?.records) ? options.records : [];
    const selected = new Set(options?.selectedIndexes || []);
    const included = options?.includeIndexes == null ? null : new Set(options.includeIndexes);
    const output = [];
    const header = String(options?.headerRaw || options?.headerFallback || "").trimEnd();
    if (header) output.push(header);
    output.push("<EOH>");
    records.forEach((record, index) => {
      if (included && !included.has(index)) return;
      const raw = typeof record === "string" ? record : String(record?.raw || "");
      const body = selected.has(index) ? ensureQslFields(raw) : raw;
      output.push(body.trim(), "<EOR>");
    });
    return `${output.join("\n")}\n`;
  }

  return {
    CONFIG_SCHEMA_VERSION,
    DEFAULT_COLUMN_SOURCES,
    MAX_ADIF_BYTES,
    MAX_ADIF_RECORDS,
    ConfigError,
    AdifError,
    normalizeMode,
    normalizeSource,
    isValidSource,
    mergeColumnSources,
    migrateAndValidateConfig,
    createSnapshotConfig,
    parseDescriptor,
    parseRecordFields,
    parseAdifDocument,
    parseADIF,
    recordToObject,
    formatAdifField,
    ensureQslFields,
    buildUpdatedAdif,
  };
});
