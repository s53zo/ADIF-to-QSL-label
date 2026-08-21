import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import fc from "fast-check";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const core = require("../../app-core.js");

const fixture = (name) => fs.readFileSync(path.join(import.meta.dirname, "../fixtures/adif", name), "utf8");

describe("ADIF 3.1.7 conformance", () => {
  it("parses headers, records, types, empty values, unknown fields, and Unicode", () => {
    const document = core.parseAdifDocument(fixture("conformance.adi"));
    expect(document.hasHeader).toBe(true);
    expect(document.headerRaw).toContain("<ADIF_VER:5:S>3.1.7");
    expect(document.records).toHaveLength(2);
    const first = core.recordToObject(document.records[0]);
    expect(first.CALL).toBe("S53Z");
    expect(first.COMMENT).toBe("memo <BAND:3>xxx");
    expect(first).not.toHaveProperty("BAND");
    expect(first.APP_TEST_NOTE).toBe("Žiga");
    expect(first.EMPTY_FIELD).toBe("");
    expect(document.records[0].fields.find((field) => field.tag === "APP_TEST_NOTE").attr).toBe("I");
  });

  it("reports a length that runs beyond the input without looping", () => {
    const document = core.parseAdifDocument(fixture("malformed-length.adi"));
    expect(document.warnings).toEqual(
      expect.arrayContaining([expect.stringMatching(/declares more characters/)]),
    );
  });

  it("does not interpret tag-like text inside a declared value", () => {
    const records = core.parseADIF("<COMMENT:16>memo <BAND:3>xxx<CALL:4>W1AW<EOR>");
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({ COMMENT: "memo <BAND:3>xxx", CALL: "W1AW" });
    expect(records[0]).not.toHaveProperty("BAND");
  });
});

describe("lossless ADIF export", () => {
  it("preserves untouched records and every untouched field", () => {
    const original = core.parseAdifDocument(fixture("conformance.adi"));
    const exported = core.buildUpdatedAdif({
      headerRaw: original.headerRaw,
      records: original.records,
      selectedIndexes: [0],
    });
    const reparsed = core.parseAdifDocument(exported);
    expect(reparsed.records).toHaveLength(original.records.length);
    const before = original.records.map(core.recordToObject);
    const after = reparsed.records.map(core.recordToObject);
    expect(after[1]).toEqual(before[1]);
    for (const [key, value] of Object.entries(before[0])) expect(after[0][key]).toBe(value);
    expect(after[0]).toMatchObject({ QSL_SENT: "Y", QSL_SENT_VIA: "B" });
    expect(reparsed.headerRaw).toContain("<PROGRAMID:8:S>QSL TEST");
  });

  it("preserves valid field values under seeded property testing", () => {
    const safeValue = fc
      .array(fc.constantFrom("A", "z", "0", " ", "_", "<", ">", ":", "Ž", "🙂"), { maxLength: 40 })
      .map((parts) => parts.join(""));
    fc.assert(
      fc.property(safeValue, safeValue, (comment, note) => {
        const source = `${core.formatAdifField("CALL", "W1AW")}${core.formatAdifField("COMMENT", comment, "S")}${core.formatAdifField("APP_TEST_NOTE", note, "I")}<EOR>`;
        const parsed = core.parseADIF(source);
        expect(parsed).toHaveLength(1);
        expect(parsed[0].COMMENT).toBe(comment.trim());
        expect(parsed[0].APP_TEST_NOTE).toBe(note.trim());
      }),
      { seed: 5302026, numRuns: 500 },
    );
  });
});
