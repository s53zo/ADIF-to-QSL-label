const AxeBuilder = require("@axe-core/playwright").default;
const { expect, test } = require("@playwright/test");

test("has no serious or critical accessibility violations", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/make_qsl_labels.html");
  await expect(page.locator("#downloadBtn")).toBeEnabled();
  const results = await new AxeBuilder({ page }).analyze();
  const violations = results.violations.filter((violation) =>
    ["serious", "critical"].includes(violation.impact),
  );
  expect(violations, JSON.stringify(violations, null, 2)).toEqual([]);
});
