# Vendored runtime dependencies

## jsPDF 4.2.1

- File: `jspdf.umd.min.js`
- Upstream: https://github.com/parallax/jsPDF
- npm package: https://registry.npmjs.org/jspdf/-/jspdf-4.2.1.tgz
- npm integrity: `sha512-YyAXyvnmjTbR4bHQRLzex3CuINCDlQnBqoSYyjJwTP2x9jDLuKDzy7aKUl0hgx3uhcl7xzg32agn5vlie6HIlQ==`
- Vendored file SHA-256: `e6551fcdc32f09d6853b2c5126d18d01d9447e0da618a41a11ebeee0f6c20d54`
- License: MIT; the upstream license notice is retained at the start of the
  vendored bundle.

The exact package version is also pinned in `package.json` and
`package-lock.json`. The application loads this local asset directly and does
not depend on a CDN for PDF generation.
