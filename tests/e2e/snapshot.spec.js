const { expect, test } = require("@playwright/test");

const labelSnapshot = {
  schemaVersion: 1,
  mode: "labels",
  pageSize: "A4",
  cols: 2,
  rows: 2,
  colOffsets: [0.5, -0.5],
  rowOffsets: [1, 2],
  columns: [
    { header: "Frequency", source: "FREQ" },
    { header: "Received", source: "RST_RCVD" },
    { header: "Custom", source: "APP_LOG_CUSTOM" },
  ],
  minColMM: [8, 8, 8],
  staticColMM: [12, 12, 12],
  dynamicCols: false,
  shrinkOnly: false,
  dbgOutline: true,
  dedupeMode: "NEWEST",
  filters: {
    dateFrom: "2026-01-01",
    dateTo: "2026-08-21",
    bands: ["20M", "40M"],
    modes: ["CW"],
    dxccCall: "JA,S53ZO",
    qslRules: { rules: [{ join: "AND", field: "QSL_SENT", op: "BLANK", value: "" }] },
    dedupeMode: "NEWEST",
  },
};

const legacyCardSnapshot = {
  mode: "qslCard",
  cols: 1,
  rows: 1,
  colOffsets: [60, 50, 30],
  rowOffsets: [20, 10, 10, 10, 0, 0, 50, 50],
  cardWidthmm: 150,
  cardHeightmm: 100,
  cardLabelWidthmm: 100,
  cardLabelHeightmm: 40,
  columns: [
    { header: "Frequency", source: "FREQ" },
    { header: "Received", source: "RST_RCVD" },
  ],
  minColMM: [8, 8],
  staticColMM: [12, 12],
};

async function loadSnapshot(page, snapshot, name = "snapshot.json") {
  await page.locator("#loadCfg").setInputFiles({
    name,
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(snapshot)),
  });
}

test.beforeEach(async ({ page }) => {
  await page.goto("/make_qsl_labels.html");
  await expect(page.locator("#downloadBtn")).toBeEnabled();
});

test("restores a legacy direct-card snapshot with a migration warning", async ({ page }) => {
  await loadSnapshot(page, legacyCardSnapshot, "legacy-card.json");
  await expect(page.locator("#configStatus")).toContainText("Snapshot restored");
  await expect(page.locator("#configStatus")).toContainText("offsets were normalized");
  await expect(page.locator('.mode-option[data-mode="qslCard"]')).toHaveClass(/selected/);
  await expect(page.locator("#colOffsets")).toHaveValue("0");
  await expect(page.locator("#rowOffsets")).toHaveValue("0");
  await expect(page.locator(".col-s").nth(0)).toHaveValue("FREQ");
  await expect(page.locator(".col-s").nth(1)).toHaveValue("RST_RCVD");
});

test("round-trips label settings and unavailable custom sources", async ({ page }) => {
  await loadSnapshot(page, labelSnapshot);
  await expect(page.locator("#configStatus")).toContainText("Snapshot restored");
  await expect(page.locator("#dynamicCols")).toHaveValue("0");
  await expect(page.locator("#shrinkOnly")).toHaveValue("0");
  await expect(page.locator("#dbgOutline")).toHaveValue("1");
  await expect(page.locator("#fDateFrom")).toHaveValue("2026-01-01");
  await expect(page.locator("#fBands")).toHaveValue("20M,40M");
  await expect(page.locator("#fDedupe")).toHaveValue("NEWEST");
  await expect(page.locator(".col-s").nth(2)).toHaveValue("APP_LOG_CUSTOM");

  await page.locator("#adifFile").setInputFiles({
    name: "different-fields.adi",
    mimeType: "text/plain",
    buffer: Buffer.from("<CALL:4>W1AW<QSO_DATE:8>20260821<MODE:2>CW<EOR>"),
  });
  await expect(page.locator(".col-s").nth(0)).toHaveValue("FREQ");
  await expect(page.locator(".col-s").nth(1)).toHaveValue("RST_RCVD");
  await expect(page.locator(".col-s").nth(2)).toHaveValue("APP_LOG_CUSTOM");

  await page.locator('div[data-section="config"] > .section-toggle').click();
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#saveCfg").click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  const saved = JSON.parse(Buffer.concat(chunks).toString("utf8"));
  expect(saved.schemaVersion).toBe(1);
  expect(saved.columns.map((column) => column.source)).toEqual(["FREQ", "RST_RCVD", "APP_LOG_CUSTOM"]);
  expect(saved).toMatchObject({ dynamicCols: false, shrinkOnly: false, dbgOutline: true });
  expect(saved.filters).toMatchObject({ bands: ["20M", "40M"], dedupeMode: "NEWEST" });
});

test("rejects unsupported future snapshots without changing mode", async ({ page }) => {
  await loadSnapshot(page, { schemaVersion: 999, mode: "qslCard" }, "future.json");
  await expect(page.locator("#configStatus")).toContainText("supports up to 1");
  await expect(page.locator('.mode-option[data-mode="labels"]')).toHaveClass(/selected/);
});

module.exports = { labelSnapshot, legacyCardSnapshot, loadSnapshot };
