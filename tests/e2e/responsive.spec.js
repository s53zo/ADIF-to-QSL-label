const { expect, test } = require("@playwright/test");

const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "tablet", width: 1024, height: 768 },
  { name: "phone", width: 390, height: 844 },
];

for (const viewport of viewports) {
  test(`${viewport.name} layout remains usable`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/make_qsl_labels.html");
    await expect(page.locator("#downloadBtn")).toBeEnabled();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
    await expect(page.locator("#preview")).toBeVisible();
    await page.locator('div[data-section="config"] > .section-toggle').click();
    await expect(page.locator("#saveCfg")).toBeVisible();
  });
}
