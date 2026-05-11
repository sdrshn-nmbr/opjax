#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24.0"]
# ///

"""
Extract Inference Engineering PDF -> structured markdown + image assets.

Personal-use local mirror. Outputs are gitignored. Re-runnable; idempotent.

Usage:
    uv run scripts/extract_pdf.py "$HOME/Downloads/Inference Engineering.pdf"
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = ROOT / "src" / "content" / "chapters"
FIGURES_DIR = ROOT / "src" / "assets" / "figures"
TOC_PATH = ROOT / "src" / "data" / "inference-toc.json"

# Heuristics
HEADER_FOOTER_BAND_PT = 50.0          # px from top/bottom counted as chrome
HEADER_FOOTER_MIN_REPEATS = 4         # text must appear on N+ pages to be chrome
MIN_FIGURE_WH_PX = 60                 # ignore decorative dots/icons
CODE_FONT_RE = re.compile(r"(Courier|Mono|Consolas|Menlo)", re.I)
BOLD_FONT_RE = re.compile(r"Bold", re.I)
ITALIC_FONT_RE = re.compile(r"(Italic|Oblique)", re.I)
HEADING_MIN_SIZE = 14.0               # font sizes >= this are likely headings
SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s+(.+)$")


# ---------- data classes ---------------------------------------------------


@dataclass
class TocEntry:
    level: int
    title: str
    page: int  # 1-indexed (PyMuPDF convention from get_toc)


@dataclass
class Chapter:
    n: int
    title: str
    slug: str
    page_start: int  # 1-indexed
    page_end: int    # 1-indexed, inclusive
    sections: list[TocEntry]


# ---------- TOC parsing ----------------------------------------------------


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60].rstrip("-") or "untitled"


def chapter_from_l2(toc: list[TocEntry], idx: int, total_pages: int) -> Chapter:
    e = toc[idx]
    next_l2 = next(
        (t for t in toc[idx + 1 :] if t.level == 2 and t.page > e.page),
        None,
    )
    page_end = (next_l2.page - 1) if next_l2 else total_pages
    sections = [
        t for t in toc[idx + 1 :]
        if t.level >= 3 and (next_l2 is None or t.page < next_l2.page)
    ]
    return Chapter(
        n=0,  # filled in by caller
        title=e.title.strip(),
        slug=slugify(e.title),
        page_start=e.page,
        page_end=page_end,
        sections=sections,
    )


def parse_toc(doc: fitz.Document) -> list[Chapter]:
    raw = doc.get_toc()
    toc = [TocEntry(level=lvl, title=title, page=page) for lvl, title, page in raw if page > 0]
    chapters: list[Chapter] = []
    skip_titles = {"table of contents", "contents", "index"}
    for i, e in enumerate(toc):
        if e.level != 2:
            continue
        if e.title.strip().lower() in skip_titles:
            continue
        chapters.append(chapter_from_l2(toc, i, doc.page_count))
    for n, ch in enumerate(chapters, start=1):
        ch.n = n
        ch.slug = f"{n:02d}-{ch.slug}"
    return chapters


# ---------- chrome (header/footer) detection -------------------------------


def detect_chrome_strings(doc: fitz.Document) -> set[str]:
    counter: Counter[str] = Counter()
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        h = page.rect.height
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            t = (text or "").strip()
            if not t or len(t) > 120:
                continue
            in_top = y1 < HEADER_FOOTER_BAND_PT
            in_bot = y0 > h - HEADER_FOOTER_BAND_PT
            if in_top or in_bot:
                counter[t] += 1
    return {t for t, c in counter.items() if c >= HEADER_FOOTER_MIN_REPEATS}


def is_page_number(text: str) -> bool:
    t = text.strip()
    return bool(t) and t.isdigit() and len(t) <= 4


# ---------- figure extraction ---------------------------------------------

CLUSTER_GAP_PT = 30.0      # vertical gap below which two image rects are one figure
FIGURE_PAD_PT = 6.0        # padding around cluster bbox before rasterizing
FIGURE_RENDER_SCALE = 2.0  # 2x = retina-ready, ~144 dpi-equivalent


def collect_image_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Return de-duplicated image rects on the page."""
    seen: set[tuple[float, float, float, float]] = set()
    rects: list[fitz.Rect] = []
    for info in page.get_images(full=True):
        xref = info[0]
        for r in page.get_image_rects(xref) or []:
            if r.width < MIN_FIGURE_WH_PX and r.height < MIN_FIGURE_WH_PX:
                continue
            key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
            if key in seen:
                continue
            seen.add(key)
            rects.append(fitz.Rect(r))
    return rects


def cluster_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    """Group rects whose vertical gap is below CLUSTER_GAP_PT into single bboxes.

    Captures composite figures where the PDF stacks several image fragments
    (chart background + sprite overlays) within one visual figure region.
    """
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r.y0, r.x0))
    clusters: list[fitz.Rect] = [fitz.Rect(rects[0])]
    for r in rects[1:]:
        last = clusters[-1]
        # If r overlaps vertically with last cluster, or sits within gap below it, merge.
        if r.y0 <= last.y1 + CLUSTER_GAP_PT:
            clusters[-1] = fitz.Rect(
                min(last.x0, r.x0),
                min(last.y0, r.y0),
                max(last.x1, r.x1),
                max(last.y1, r.y1),
            )
        else:
            clusters.append(fitz.Rect(r))
    return clusters


def extract_figures(doc: fitz.Document, page: fitz.Page, pno_1: int) -> list[tuple[float, str]]:
    """Render each image-cluster on the page as a single PNG, including any
    text overlays the PDF composites on top (axis labels, badges, captions
    drawn inside the figure region). Returns (y_top, md_ref) tuples."""
    out: list[tuple[float, str]] = []
    raw_rects = collect_image_rects(page)
    clusters = cluster_rects(raw_rects)
    if not clusters:
        return out
    page_rect = page.rect
    matrix = fitz.Matrix(FIGURE_RENDER_SCALE, FIGURE_RENDER_SCALE)
    for idx, cluster in enumerate(clusters, start=1):
        # Pad and clamp to page bounds. Widen slightly horizontally so labels
        # rendered as text adjacent to the raster get included.
        clip = fitz.Rect(
            max(page_rect.x0, cluster.x0 - FIGURE_PAD_PT * 2),
            max(page_rect.y0, cluster.y0 - FIGURE_PAD_PT),
            min(page_rect.x1, cluster.x1 + FIGURE_PAD_PT * 2),
            min(page_rect.y1, cluster.y1 + FIGURE_PAD_PT),
        )
        try:
            pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        except Exception as exc:
            print(f"  ! pixmap render failed p{pno_1} fig{idx}: {exc}", file=sys.stderr)
            continue
        fname = f"p{pno_1:03d}-fig{idx:02d}.png"
        pix.save(FIGURES_DIR / fname)
        ref = f"![](../../assets/figures/{fname})"
        out.append((cluster.y0, ref))
    return out


# ---------- text extraction per page --------------------------------------


def block_kind(spans: list[dict]) -> str:
    """heading | code | body — based on font/size of the dominant span."""
    if not spans:
        return "body"
    sizes = [s["size"] for s in spans]
    fonts = [s.get("font", "") for s in spans]
    max_size = max(sizes)
    any_code = any(CODE_FONT_RE.search(f) for f in fonts)
    if any_code and not any(s["size"] >= HEADING_MIN_SIZE for s in spans):
        return "code"
    if max_size >= HEADING_MIN_SIZE:
        return "heading"
    return "body"


def render_inline(spans: list[dict]) -> str:
    """Concatenate spans into plain text with **bold** / *italic* markers preserved."""
    parts: list[str] = []
    for s in spans:
        text = s.get("text", "")
        if not text:
            continue
        font = s.get("font", "")
        bold = bool(BOLD_FONT_RE.search(font))
        italic = bool(ITALIC_FONT_RE.search(font))
        # Don't bold/italicise pure whitespace or punctuation
        wraps = ""
        if text.strip():
            if bold and italic:
                wraps = "***"
            elif bold:
                wraps = "**"
            elif italic:
                wraps = "*"
        if wraps:
            parts.append(f"{wraps}{text}{wraps}")
        else:
            parts.append(text)
    return "".join(parts)


def fix_hyphenation(text: str) -> str:
    # "infer-\nence" -> "inference"; "real- world" leave alone.
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def normalise_paragraph(text: str) -> str:
    text = fix_hyphenation(text)
    text = re.sub(r"\s*\n\s*", " ", text)  # join soft wraps within paragraph
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def page_to_markdown(
    page: fitz.Page,
    pno_1: int,
    chrome: set[str],
    section_anchors: dict[int, list[str]],
    figures_by_page: list[tuple[float, str]],
) -> list[str]:
    """Convert one page into a list of markdown blocks (in reading order)."""
    out: list[str] = []
    # Inject any TOC-driven section headings declared as starting on this page.
    for header_md in section_anchors.get(pno_1, []):
        out.append(header_md)

    h = page.rect.height
    raw = page.get_text("dict")
    items: list[tuple[float, str]] = []  # (y_top, md_block)

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0=text, 1=image
            continue
        bbox = block.get("bbox", (0, 0, 0, 0))
        y0, y1 = bbox[1], bbox[3]
        # skip header/footer band
        if y1 < HEADER_FOOTER_BAND_PT or y0 > h - HEADER_FOOTER_BAND_PT:
            continue
        # collect spans in line order; preserve line breaks for code, flatten for body
        lines = block.get("lines", [])
        if not lines:
            continue
        all_spans: list[dict] = []
        for ln in lines:
            all_spans.extend(ln.get("spans", []))
        if not all_spans:
            continue
        text_concat = "".join(s.get("text", "") for s in all_spans).strip()
        if not text_concat:
            continue
        if text_concat in chrome or is_page_number(text_concat):
            continue

        kind = block_kind(all_spans)

        if kind == "code":
            code_lines = []
            for ln in lines:
                code_lines.append("".join(s.get("text", "") for s in ln.get("spans", [])).rstrip())
            code = "\n".join(code_lines).rstrip()
            md = "```\n" + code + "\n```"
            items.append((y0, md))
            continue

        if kind == "heading":
            text = normalise_paragraph("".join(s.get("text", "") for s in all_spans))
            if not text:
                continue
            # Decide heading depth: if it matches "1.2.3 …" use depth+2, else default to ##
            m = SECTION_NUM_RE.match(text)
            depth = 2
            if m:
                depth = min(2 + m.group(1).count("."), 4)
            md = ("#" * depth) + " " + text
            items.append((y0, md))
            continue

        # body paragraph: preserve per-line boundaries so hyphenation regex
        # can rejoin "infer-\nence" across the soft wrap before flattening.
        per_line = [render_inline(ln.get("spans", [])) for ln in lines]
        rendered = "\n".join(per_line)
        text = normalise_paragraph(rendered)
        if not text:
            continue
        items.append((y0, text))

    # merge in figures by their y position
    items.extend(figures_by_page)
    items.sort(key=lambda t: t[0])
    out.extend(md for _y, md in items)
    return out


# ---------- chapter assembly ----------------------------------------------


def chapter_section_anchors(ch: Chapter) -> dict[int, list[str]]:
    """Map page -> [headings to inject at top of that page] from TOC L3/L4 entries."""
    out: dict[int, list[str]] = defaultdict(list)
    for s in ch.sections:
        depth = min(s.level, 4)  # L3 -> ## (depth 2), L4 -> ### (depth 3)
        # Map: L3 -> ##, L4 -> ###, L5 -> ####
        md_depth = depth - 1
        out[s.page].append(("#" * md_depth) + " " + s.title.strip())
    return out


def assemble_chapter(doc: fitz.Document, ch: Chapter, chrome: set[str]) -> str:
    anchors = chapter_section_anchors(ch)
    body_blocks: list[str] = []
    for pno_1 in range(ch.page_start, ch.page_end + 1):
        page = doc.load_page(pno_1 - 1)
        figures = extract_figures(doc, page, pno_1)
        body_blocks.extend(page_to_markdown(page, pno_1, chrome, anchors, figures))
    body = "\n\n".join(b for b in body_blocks if b.strip())
    body = re.sub(r"\n{3,}", "\n\n", body)
    front = (
        "---\n"
        f"title: {json.dumps(ch.title)}\n"
        f"chapter: {ch.n}\n"
        f"pageStart: {ch.page_start}\n"
        f"pageEnd: {ch.page_end}\n"
        "---\n\n"
        f"# {ch.title}\n\n"
    )
    return front + body + "\n"


# ---------- driver ---------------------------------------------------------


def reset_output_dirs() -> None:
    if CHAPTERS_DIR.exists():
        shutil.rmtree(CHAPTERS_DIR)
    if FIGURES_DIR.exists():
        shutil.rmtree(FIGURES_DIR)
    CHAPTERS_DIR.mkdir(parents=True)
    FIGURES_DIR.mkdir(parents=True)
    TOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    args = ap.parse_args()
    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    print(f"opening {args.pdf}")
    doc = fitz.open(args.pdf)
    print(f"  {doc.page_count} pages")

    chapters = parse_toc(doc)
    if not chapters:
        print("no L2 TOC entries found — cannot extract chapters", file=sys.stderr)
        return 3
    print(f"  {len(chapters)} chapters detected")

    print("scanning for header/footer chrome…")
    chrome = detect_chrome_strings(doc)
    print(f"  {len(chrome)} repeated chrome strings filtered")

    reset_output_dirs()

    toc_meta = []
    for ch in chapters:
        md = assemble_chapter(doc, ch, chrome)
        out_path = CHAPTERS_DIR / f"{ch.slug}.md"
        out_path.write_text(md, encoding="utf-8")
        toc_meta.append({
            "n": ch.n,
            "title": ch.title,
            "slug": ch.slug,
            "pageStart": ch.page_start,
            "pageEnd": ch.page_end,
            "sectionCount": len(ch.sections),
        })
        n_figs = len(list(FIGURES_DIR.glob(f"p{ch.page_start:03d}-*"))) + sum(
            len(list(FIGURES_DIR.glob(f"p{p:03d}-*")))
            for p in range(ch.page_start + 1, ch.page_end + 1)
        )
        print(f"  ch{ch.n:02d} pp{ch.page_start}-{ch.page_end}  {ch.title[:60]}  ({n_figs} figs)")

    TOC_PATH.write_text(json.dumps(toc_meta, indent=2), encoding="utf-8")
    total_figs = len(list(FIGURES_DIR.glob("*")))
    print(f"done: {len(chapters)} chapters, {total_figs} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
