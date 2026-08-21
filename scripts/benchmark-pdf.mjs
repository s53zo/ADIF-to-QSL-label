import { spawn, execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "@playwright/test";

const repository = path.resolve(import.meta.dirname, "..");
const baselineRef = process.env.PDF_BASELINE_REF || "HEAD";
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "qsl-pdf-benchmark-"));
const archive = path.join(temporary, "baseline.tar");
const baseline = path.join(temporary, "baseline");
fs.mkdirSync(baseline);
execFileSync("git", ["archive", "--format=tar", baselineRef, "-o", archive], { cwd: repository });
execFileSync("tar", ["-xf", archive, "-C", baseline]);

const servers = [];
function serve(directory, port) {
  const process = spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
    cwd: directory,
    stdio: "ignore",
  });
  servers.push(process);
  return `http://127.0.0.1:${port}/make_qsl_labels.html`;
}

async function ready(url) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      // The server may still be starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Server did not start: ${url}`);
}

async function measure(browser, name, url) {
  const page = await browser.newPage();
  await page.route(/^https?:\/(?!\/127\.0\.0\.1:)/, (route) => route.fulfill({ status: 204, body: "" }));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
  await page.waitForFunction(() => !document.querySelector("#downloadBtn").disabled && window.jspdf?.jsPDF);
  await page.evaluate(() => {
    window.__pdfBenchmark = { last: performance.now(), maxGap: 0, ticks: 0 };
    window.__pdfBenchmark.timer = setInterval(() => {
      const now = performance.now();
      window.__pdfBenchmark.maxGap = Math.max(window.__pdfBenchmark.maxGap, now - window.__pdfBenchmark.last);
      window.__pdfBenchmark.last = now;
      window.__pdfBenchmark.ticks += 1;
    }, 16);
  });
  const started = performance.now();
  const downloadPromise = page.waitForEvent("download");
  await page.evaluate(() => document.querySelector("#downloadBtn").click());
  const download = await downloadPromise;
  const target = path.join(temporary, `${name}.pdf`);
  await download.saveAs(target);
  await page.waitForTimeout(50);
  const responsiveness = await page.evaluate(() => {
    clearInterval(window.__pdfBenchmark.timer);
    return { maxEventLoopGapMs: window.__pdfBenchmark.maxGap, ticks: window.__pdfBenchmark.ticks };
  });
  const result = {
    name,
    durationMs: Math.round(performance.now() - started),
    bytes: fs.statSync(target).size,
    maxEventLoopGapMs: Math.round(responsiveness.maxEventLoopGapMs),
    timerTicks: responsiveness.ticks,
  };
  await page.close();
  return result;
}

const currentUrl = serve(repository, 4180);
const baselineUrl = serve(baseline, 4181);
let browser;
try {
  await Promise.all([ready(currentUrl), ready(baselineUrl)]);
  browser = await chromium.launch({ headless: true });
  const results = [];
  results.push(await measure(browser, `${baselineRef}-baseline`, baselineUrl));
  results.push(await measure(browser, "working-tree", currentUrl));
  console.table(results);
} finally {
  if (browser) await browser.close();
  servers.forEach((server) => server.kill());
  fs.rmSync(temporary, { recursive: true, force: true });
}
