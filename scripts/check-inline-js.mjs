import fs from "node:fs";
import { parse } from "espree";

const html = fs.readFileSync(new URL("../make_qsl_labels.html", import.meta.url), "utf8");
const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)];

if (!scripts.length) throw new Error("No inline application scripts found.");

scripts.forEach((match, index) => {
  try {
    parse(match[1], { ecmaVersion: "latest", sourceType: "script" });
  } catch (error) {
    throw new Error(`Inline script ${index + 1} has invalid JavaScript: ${error.message}`, { cause: error });
  }
});

console.log(`Validated ${scripts.length} inline script${scripts.length === 1 ? "" : "s"}.`);
