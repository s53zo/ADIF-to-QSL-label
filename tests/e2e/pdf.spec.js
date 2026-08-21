const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const { PDFDocument } = require("pdf-lib");
const pixelmatch = require("pixelmatch").default;
const { PNG } = require("pngjs");
const { expect, test } = require("@playwright/test");

async function exportPdf(page) {
  const downloadPromise = page.waitForEvent("download");
  await page.locator("#downloadBtn").click();
  await expect(page.locator("#pdfProgress")).toHaveClass(/is-active/);
  const download = await downloadPromise;
  const target = test.info().outputPath(download.suggestedFilename());
  await download.saveAs(target);
  return target;
}

async function expectPageSize(pdfPath, widthMm, heightMm, pageCount) {
  const pdf = await PDFDocument.load(fs.readFileSync(pdfPath));
  expect(pdf.getPageCount()).toBe(pageCount);
  const size = pdf.getPage(0).getSize();
  const toMm = (points) => (points * 25.4) / 72;
  expect(Math.abs(toMm(size.width) - widthMm)).toBeLessThanOrEqual(0.1);
  expect(Math.abs(toMm(size.height) - heightMm)).toBeLessThanOrEqual(0.1);
}

async function capturePreview(page) {
  await page.evaluate(async () => {
    await document.fonts.ready;
    window.recompute();
  });
  return page.locator("#preview").evaluate((canvas) => ({
    width: canvas.width,
    height: canvas.height,
    png: canvas.toDataURL("image/png").split(",")[1],
  }));
}

function expectPdfMatchesPreview(pdfPath, preview, artifactPrefix) {
  const previewPath = test.info().outputPath(`${artifactPrefix}-preview.png`);
  const renderedBase = test.info().outputPath(`${artifactPrefix}-pdf-page`);
  fs.writeFileSync(previewPath, Buffer.from(preview.png, "base64"));
  execFileSync("pdftoppm", [
    "-f",
    "1",
    "-singlefile",
    "-png",
    "-scale-to-x",
    String(preview.width),
    "-scale-to-y",
    String(preview.height),
    pdfPath,
    renderedBase,
  ]);
  const previewPng = PNG.sync.read(fs.readFileSync(previewPath));
  const pdfPng = PNG.sync.read(fs.readFileSync(`${renderedBase}.png`));
  expect(pdfPng.width).toBe(previewPng.width);
  expect(pdfPng.height).toBe(previewPng.height);
  const diff = new PNG({ width: previewPng.width, height: previewPng.height });
  const changed = pixelmatch(previewPng.data, pdfPng.data, diff.data, previewPng.width, previewPng.height, {
    threshold: 0.15,
  });
  fs.writeFileSync(test.info().outputPath(`${artifactPrefix}-preview-pdf-diff.png`), PNG.sync.write(diff));
  // PDF rasterization re-antialiases the 4x source image. A 6% changed-pixel
  // ceiling catches layout shifts while allowing those edge-only differences.
  expect(changed / (previewPng.width * previewPng.height)).toBeLessThan(0.06);
}

test.beforeEach(async ({ page }) => {
  await page.goto("/make_qsl_labels.html");
  await expect(page.locator("#downloadBtn")).toBeEnabled();
});

test("exports a correctly sized multi-page A4 PDF matching the preview", async ({ page }) => {
  const preview = await capturePreview(page);
  const pdfPath = await exportPdf(page);
  await expectPageSize(pdfPath, 210, 297, 2);
  expectPdfMatchesPreview(pdfPath, preview, "labels");
});

test("exports an exact landscape direct-card page", async ({ page }) => {
  const snapshot = {
    mode: "qslCard",
    cols: 1,
    rows: 1,
    colOffsets: [1, 2, 3],
    rowOffsets: [1, 2],
    cardWidthmm: 150,
    cardHeightmm: 100,
    cardLabelWidthmm: 100,
    cardLabelHeightmm: 40,
    columns: [{ header: "Date", source: "DATE" }],
    minColMM: [8],
    staticColMM: [12],
  };
  await page.locator("#loadCfg").setInputFiles({
    name: "card.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(snapshot)),
  });
  await expect(page.locator("#configStatus")).toContainText("Snapshot restored");
  await page.locator("#adifFile").setInputFiles({
    name: "one.adi",
    mimeType: "text/plain",
    buffer: Buffer.from("<CALL:4>W1AW<QSO_DATE:8>20260821<TIME_ON:4>1200<BAND:3>20M<MODE:2>CW<EOR>"),
  });
  await expect(page.locator("#banner")).toBeHidden();
  const preview = await capturePreview(page);
  const pdfPath = await exportPdf(page);
  await expectPageSize(pdfPath, 150, 100, 1);
  expectPdfMatchesPreview(pdfPath, preview, "card");
});

test("can cancel an export without downloading a partial PDF", async ({ page }) => {
  let downloaded = false;
  page.on("download", () => {
    downloaded = true;
  });
  const records = Array.from({ length: 500 }, (_, index) => {
    const call = `T${String(index).padStart(4, "0")}`;
    return `<CALL:${call.length}>${call}<QSO_DATE:8>20260821<TIME_ON:4>1200<BAND:3>20M<MODE:2>CW<EOR>`;
  }).join("\n");
  await page.locator("#adifFile").setInputFiles({
    name: "many.adi",
    mimeType: "text/plain",
    buffer: Buffer.from(records),
  });
  await page.locator("#downloadBtn").evaluate((button) => button.click());
  await expect(page.locator("#pdfProgress")).toHaveClass(/is-active/);
  await page.locator("#cancelPdf").evaluate((button) => button.click());
  await expect(page.locator("#pdfProgressText")).toContainText("cancelled");
  await page.waitForTimeout(300);
  expect(downloaded).toBe(false);
  await expect(page.locator("#downloadBtn")).toBeEnabled();
});
