import { createRequire } from "node:module";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const core = require("../../app-core.js");

function legacyCard() {
  return {
    mode: "qslCard",
    cols: 1,
    rows: 1,
    colOffsets: [60, 50, 30],
    rowOffsets: [20, 10, 10, 10, 0, 0, 50, 50],
    columns: [
      { header: "Frequency", source: "FREQ" },
      { header: "Received", source: "RST_RCVD" },
    ],
    minColMM: [8, 8],
    staticColMM: [12, 12],
    dynamicCols: true,
    shrinkOnly: true,
    dbgOutline: true,
  };
}

describe("versioned snapshot configuration", () => {
  it("migrates the legacy direct-card offset mismatch", () => {
    const result = core.migrateAndValidateConfig(legacyCard());
    expect(result.migrated).toBe(true);
    expect(result.value).toMatchObject({
      schemaVersion: 1,
      mode: "qslCard",
      cols: 1,
      rows: 1,
      colOffsets: [0],
      rowOffsets: [0],
    });
    expect(result.warnings).toContain("Legacy card-only grid offsets were normalized.");
  });

  it("does not hide a malformed legacy label grid", () => {
    expect(() =>
      core.migrateAndValidateConfig({
        mode: "labels",
        cols: 3,
        rows: 8,
        colOffsets: [0],
        rowOffsets: Array(8).fill(0),
      }),
    ).toThrow(/colOffsets length must match cols/);
  });

  it("rejects future schema versions with an actionable error", () => {
    expect(() => core.migrateAndValidateConfig({ schemaVersion: 99 })).toThrow(/supports up to 1/);
  });

  it("rejects invalid custom ADIF source names", () => {
    expect(() =>
      core.migrateAndValidateConfig({
        schemaVersion: 1,
        columns: [{ header: "Bad", source: "<script>" }],
      }),
    ).toThrow(/Invalid ADIF column source/);
  });

  it("drops prototype-pollution keys while preserving safe data", () => {
    const input = JSON.parse('{"mode":"labels","__proto__":{"polluted":true}}');
    const result = core.migrateAndValidateConfig(input);
    expect(result.value.mode).toBe("labels");
    expect({}.polluted).toBeUndefined();
    expect(Object.prototype.hasOwnProperty.call(result.value, "__proto__")).toBe(false);
  });

  it("round-trips all serializable card settings canonically", () => {
    const migrated = core.migrateAndValidateConfig(legacyCard()).value;
    const snapshot = core.createSnapshotConfig(migrated, {
      dateFrom: "2026-01-01",
      dateTo: "2026-08-21",
      bands: ["20M", "40M"],
      modes: ["CW"],
      dxccCall: "JA,S53ZO",
      qslRules: { rules: [{ join: "AND", field: "QSL_SENT", op: "BLANK", value: "" }] },
      dedupeMode: "NEWEST",
    });
    const restored = core.migrateAndValidateConfig(JSON.parse(JSON.stringify(snapshot))).value;
    expect(restored).toEqual(snapshot);
    expect(restored.dynamicCols).toBe(true);
    expect(restored.shrinkOnly).toBe(true);
    expect(restored.dbgOutline).toBe(true);
    expect(restored.filters.dedupeMode).toBe("NEWEST");
  });
});

describe("custom column sources", () => {
  it("keeps saved sources that are absent from the loaded ADIF", () => {
    expect(core.mergeColumnSources(["freq", "RST_RCVD", "APP_LOG_CUSTOM"], ["CALL"])).toEqual(
      expect.arrayContaining(["FREQ", "RST_RCVD", "APP_LOG_CUSTOM", "CALL"]),
    );
  });

  it("does not replace saved sources when the discovered data set changes", () => {
    const first = core.mergeColumnSources(["RST_RCVD"], ["CALL", "BAND"]);
    const second = core.mergeColumnSources([first.find((source) => source === "RST_RCVD")], ["CALL", "FREQ"]);
    expect(second).toContain("RST_RCVD");
  });
});
