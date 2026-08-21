const js = require("@eslint/js");

module.exports = [
  {
    ignores: ["node_modules/**", "test-results/**", "playwright-report/**", "vendor/**"],
  },
  js.configs.recommended,
  {
    files: ["app-core.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        globalThis: "readonly",
        module: "readonly",
      },
    },
  },
  {
    files: ["tests/e2e/**/*.js", "playwright.config.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: {
        Buffer: "readonly",
        __dirname: "readonly",
        process: "readonly",
        require: "readonly",
        module: "readonly",
        document: "readonly",
        window: "readonly",
      },
    },
  },
  {
    files: ["tests/unit/**/*.mjs", "scripts/**/*.mjs"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        console: "readonly",
        URL: "readonly",
        fetch: "readonly",
        setTimeout: "readonly",
        setInterval: "readonly",
        clearInterval: "readonly",
        document: "readonly",
        window: "readonly",
        performance: "readonly",
        process: "readonly",
      },
    },
  },
];
