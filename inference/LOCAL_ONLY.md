# Local-only

This project is a personal reading surface for *Inference Engineering* by Philip Kiely. The user owns the PDF and uses this site as a nicer local reader.

**Rules:**
- The site is never deployed and never pushed to a public host.
- `src/content/chapters/`, `src/assets/figures/`, and `src/data/inference-toc.json` are gitignored — they hold extracted verbatim content from the PDF and must never be committed.
- The PDF itself is gitignored.
- Only the scaffold (Astro config, layouts, components, styles, extraction script) lives in version control.

To regenerate the book contents on a fresh machine:

```bash
pnpm install
pnpm run extract        # reads ~/Downloads/Inference Engineering.pdf
pnpm run dev            # http://localhost:4322
```
