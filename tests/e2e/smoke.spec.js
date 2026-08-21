const { expect, test } = require("@playwright/test");

test("loads offline runtime assets without unexpected console errors", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.route(/^https?:\/(?!\/127\.0\.0\.1:4173)/, (route) => route.fulfill({ status: 204, body: "" }));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/make_qsl_labels.html");
  await expect(page.locator("#downloadBtn")).toBeEnabled();
  expect(await page.evaluate(() => window.jspdf?.jsPDF?.version)).toBe("4.2.1");
  expect(errors).toEqual([]);
});
