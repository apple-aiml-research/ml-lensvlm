# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
text-to-image renderer using PIL.
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .rendering_config import RenderConfig


# ── Text wrapping ──────────────────────────────────────────────────────────

def _text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def wrap_text_to_lines(text: str, font: ImageFont.FreeTypeFont, width: int, margin: int) -> List[str]:
    """Wrap text into lines using simple word wrapping."""
    max_width = max(1, width - 2 * margin)
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []

    for raw_line in clean_text.split("\n"):
        if raw_line.strip() == "":
            lines.append("")
            continue

        words = raw_line.split(" ")
        current: List[str] = []
        for word in words:
            candidate = " ".join(current + [word]) if current else word
            if _text_width(candidate, font) <= max_width:
                current.append(word)
                continue

            if current:
                lines.append(" ".join(current))
            elif _text_width(word, font) > max_width:
                chunk = word
                while chunk:
                    cutoff = len(chunk)
                    while cutoff > 0 and _text_width(chunk[:cutoff], font) > max_width:
                        cutoff -= 1
                    if cutoff == 0:
                        cutoff = 1
                    lines.append(chunk[:cutoff])
                    chunk = chunk[cutoff:]
                current = []
                continue

            current = [word]

        if current:
            lines.append(" ".join(current))

    return lines

# Bold font for page labels
_LABEL_FONT_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "fonts", "DejaVuSans-Bold.ttf"),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Label dimensions (conservative estimate for collision check)
def _get_label_font(size: int = 10) -> ImageFont.FreeTypeFont:
    for path in _LABEL_FONT_CANDIDATES:
        path = os.path.abspath(path)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _add_page_label(img: Image.Image, page_num: int) -> Image.Image:
    """Add 'P{n}' label to bottom-right corner."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = _get_label_font(10)

    text = f"{page_num}"
    bbox = draw.textbbox((0, 0), text, font=font, anchor="mm")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    label_w = text_w + 4
    label_h = text_h + 2
    offset = 1

    rect_x2 = img.width - offset
    rect_y2 = img.height - offset
    rect_x1 = rect_x2 - label_w
    rect_y1 = rect_y2 - label_h

    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill="white", outline="black")
    center_x = (rect_x1 + rect_x2) / 2
    center_y = (rect_y1 + rect_y2) / 2
    draw.text((center_x, center_y), text, fill="black", font=font, anchor="mm")

    return img


@dataclass
class PageRenderResult:
    """Result of rendering text into multiple pages."""
    pages: List[Image.Image]
    page_texts: List[str]
    num_pages: int
    config_key: str
    # 1-indexed page numbers containing evidence. Empty if no evidence_spans
    # were passed to render_pages, or if none survived rendering.
    evidence_pages: List[int] = field(default_factory=list)


# ── Text normalization ─────────────────────────────

def normalize_text_for_rendering(text: str) -> str:
    """Normalize text before rendering to eliminate whitespace waste.

    Normalizes whitespace before rendering:
    1. Split into paragraphs on double newlines
    2. Normalize each paragraph (collapse internal newlines/spaces)
    3. Merge short paragraphs (< 80 chars) into neighbors
    4. Rejoin with single spaces (no paragraph breaks in rendered output)
    """
    raw_paragraphs = re.split(r'\n\s*\n', text)

    paragraphs = [_normalize_paragraph(p) for p in raw_paragraphs]
    paragraphs = [p for p in paragraphs if p]

    if not paragraphs:
        return ""

    paragraphs = _merge_short_paragraphs(paragraphs)

    return ' '.join(paragraphs)


def normalize_text_for_rendering_with_map(text: str) -> Tuple[str, List[int]]:
    """Same transformation as ``normalize_text_for_rendering`` but also returns
    a position map.

    Returns:
        (normalized_text, pos_map) where ``pos_map[i]`` is the index in
        ``normalized_text`` corresponding to the original character ``text[i]``,
        or -1 if that character was dropped (collapsed leading/trailing
        whitespace, or whitespace beyond the first in a run).

    The map is used by ``render_pages`` to remap ``evidence_spans`` from
    original-text coordinates into normalized-text coordinates so the
    renderer's deterministic line-overlap math gives correct ``evidence_pages``.
    """
    pos_map: List[int] = [-1] * len(text)
    if not text:
        return "", pos_map

    # Step A: split into paragraphs on \n\s*\n; track each paragraph's start
    # offset in the original text.
    raw_paras: List[Tuple[int, str]] = []  # (orig_start, raw_text)
    pos = 0
    for m in re.finditer(r'\n\s*\n', text):
        if pos < m.start():
            raw_paras.append((pos, text[pos:m.start()]))
        pos = m.end()
    if pos < len(text):
        raw_paras.append((pos, text[pos:]))

    # Step B: normalize each paragraph at the character level. Whitespace runs
    # collapse to single space; leading/trailing whitespace is stripped.
    # Build (orig_idx, position-in-this-paragraph's-normalized-text) pairs.
    def _normalize_para(orig_start: int, raw: str) -> Tuple[str, List[Tuple[int, int]]]:
        out: List[str] = []
        pairs: List[Tuple[int, int]] = []  # (orig_idx, local_norm_idx or -1)
        prev_space = True  # leading whitespace gets stripped
        for i, c in enumerate(raw):
            if c.isspace():
                if prev_space:
                    pairs.append((orig_start + i, -1))
                else:
                    pairs.append((orig_start + i, len(out)))
                    out.append(' ')
                    prev_space = True
            else:
                pairs.append((orig_start + i, len(out)))
                out.append(c)
                prev_space = False
        # Strip trailing space (re-map any orig chars that targeted it)
        while out and out[-1] == ' ':
            trailing = len(out) - 1
            out.pop()
            pairs = [(o, -1 if n == trailing else n) for o, n in pairs]
        return ''.join(out), pairs

    norm_paras: List[Tuple[str, List[Tuple[int, int]]]] = []
    for orig_start, raw in raw_paras:
        norm, pairs = _normalize_para(orig_start, raw)
        if norm:
            norm_paras.append((norm, pairs))

    if not norm_paras:
        return "", pos_map

    # Step C: merge short paragraphs (<80 chars) into the previous one with
    # ' ' separator. Mirror the exact logic of _merge_short_paragraphs.
    def _shift_pairs(pairs, shift):
        return [(o, n + shift if n >= 0 else -1) for o, n in pairs]

    merged: List[Tuple[str, List[Tuple[int, int]]]] = []
    for norm, pairs in norm_paras:
        if merged and len(norm) < 80:
            prev_norm, prev_pairs = merged[-1]
            shift = len(prev_norm) + 1  # +1 for the ' ' separator
            merged[-1] = (prev_norm + ' ' + norm, prev_pairs + _shift_pairs(pairs, shift))
        else:
            merged.append((norm, list(pairs)))
    if len(merged) > 1 and len(merged[0][0]) < 80:
        n0, p0 = merged[0]
        n1, p1 = merged[1]
        shift = len(n0) + 1
        merged[1] = (n0 + ' ' + n1, p0 + _shift_pairs(p1, shift))
        merged.pop(0)

    # Step D: join all merged paragraphs with ' '
    final_parts: List[str] = []
    cumulative = 0
    for i, (norm, pairs) in enumerate(merged):
        if i > 0:
            cumulative += 1  # for the joining ' '
            final_parts.append(' ')
        final_parts.append(norm)
        for orig_idx, n in pairs:
            if 0 <= orig_idx < len(text):
                pos_map[orig_idx] = (n + cumulative) if n >= 0 else -1
        cumulative += len(norm)

    return ''.join(final_parts), pos_map


def _remap_evidence_spans(
    evidence_spans: List[Tuple[int, int]],
    pos_map: List[int],
    norm_len: int,
) -> List[Tuple[int, int]]:
    """Remap (start, end) char spans from original to normalized coordinates.

    For each span, find the smallest non-negative pos_map value within
    [start, end) — that's the normalized start. Find the largest non-negative
    pos_map value within [start, end), +1 — that's the normalized end.
    Spans whose chars all collapse to whitespace (no surviving normalized
    chars) are dropped.
    """
    out: List[Tuple[int, int]] = []
    for s, e in evidence_spans:
        s = max(0, s)
        e = min(len(pos_map), e)
        if s >= e:
            continue
        first = -1
        last = -1
        for i in range(s, e):
            n = pos_map[i]
            if n >= 0:
                if first < 0:
                    first = n
                last = n
        if first < 0:
            continue
        out.append((first, min(last + 1, norm_len)))
    return out


def _normalize_paragraph(text: str) -> str:
    """Normalize a paragraph by collapsing internal newlines and whitespace to spaces."""
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _merge_short_paragraphs(paragraphs: List[str], min_chars: int = 80) -> List[str]:
    """Merge paragraphs shorter than min_chars into their neighbors."""
    if not paragraphs:
        return []

    merged: List[str] = []
    for para in paragraphs:
        if merged and len(para) < min_chars:
            merged[-1] = merged[-1] + ' ' + para
        else:
            merged.append(para)
    if len(merged) > 1 and len(merged[0]) < min_chars:
        merged[1] = merged[0] + ' ' + merged[1]
        merged.pop(0)
    return merged


def render_pages(
    text: str,
    config: RenderConfig,
    evidence_spans: Optional[List[Tuple[int, int]]] = None,
) -> Optional[PageRenderResult]:
    """Render text into content-tight page images.

    Approach:
    1. Normalize text (collapse paragraph breaks, merge short fragments)
    2. Wrap text into lines
    3. Clear label zones (shift last word if it would collide with page label)
    4. Render one tall image with the full text
    5. Split into pages by target_lines_per_page (~500-1000 tokens per page)
    6. Crop content-tight pages from the tall image
    7. Add page number labels

    Args:
        text: The text to render.
        config: RenderConfig controlling visual parameters.
        evidence_spans: Optional list of (start, end) char offsets in the
            ORIGINAL text. When provided, the renderer threads these
            through normalization (via a position map) and the line-wrap,
            then computes ``evidence_pages`` deterministically as the
            1-indexed pages each span lands on. No string matching.

    Returns:
        PageRenderResult or None if text is empty.
    """
    if not text or not text.strip():
        return None

    # Step 1: Normalize text. If evidence_spans was provided, also build a
    # position map (orig→normalized) so we can remap the spans.
    if evidence_spans is not None:
        normalized, pos_map = normalize_text_for_rendering_with_map(text)
    else:
        normalized = normalize_text_for_rendering(text)
        pos_map = None
    if not normalized:
        return None

    font = ImageFont.truetype(config.font_path, config.font_size)
    line_height = config.line_height
    target_lpp = config.target_lines_per_page()

    # Step 2: Wrap text into lines
    lines = wrap_text_to_lines(normalized, font, config.width, config.margin_px)

    # Strip trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None

    # If we have evidence_spans, compute the (line_first, line_last) range
    # each remapped span covers. This uses the same offset arithmetic
    # `wrap_text_to_lines` produces — text wrapping preserves character
    # positions modulo the spaces that became newlines, so a remapped
    # offset O in `normalized` lands on the line whose cumulative-char count
    # crosses O.
    evidence_line_ranges: List[Tuple[int, int]] = []
    if evidence_spans is not None and pos_map is not None:
        remapped = _remap_evidence_spans(evidence_spans, pos_map, len(normalized))
        # Build line-end offsets. wrap_text_to_lines splits `normalized` at
        # word boundaries; some spaces become line breaks (1 space → 1 newline,
        # same length), and very long words may be character-wrapped (extra
        # newlines without consuming a source char). To map char offsets in
        # `normalized` to line indices we walk through the lines and consume
        # `normalized` characters one-by-one, tracking which line each char
        # ends up on.
        char_to_line: List[int] = [0] * len(normalized)
        norm_idx = 0
        for line_idx, line_text in enumerate(lines):
            for c in line_text:
                # Skip any spaces in `normalized` that aren't on this line
                # (i.e., spaces that became line breaks before this position).
                while norm_idx < len(normalized) and normalized[norm_idx] != c and normalized[norm_idx].isspace():
                    char_to_line[norm_idx] = line_idx
                    norm_idx += 1
                if norm_idx < len(normalized) and normalized[norm_idx] == c:
                    char_to_line[norm_idx] = line_idx
                    norm_idx += 1
        # Any remaining chars (trailing whitespace) go on the last line
        last_line = max(0, len(lines) - 1)
        while norm_idx < len(normalized):
            char_to_line[norm_idx] = last_line
            norm_idx += 1

        for s, e in remapped:
            if s >= len(char_to_line) or e <= 0:
                continue
            s_clamped = max(0, min(s, len(char_to_line) - 1))
            e_clamped = max(0, min(e - 1, len(char_to_line) - 1))
            evidence_line_ranges.append((char_to_line[s_clamped], char_to_line[e_clamped]))

    # Step 3: Render one tall image (always left-aligned)
    total_height = config.margin_px * 2 + line_height * len(lines)
    if total_height <= 0:
        total_height = config.margin_px * 2 + line_height

    tall_img = Image.new("RGB", (config.width, total_height), config.bg_color)
    draw = ImageDraw.Draw(tall_img)

    y = config.margin_px
    for line in lines:
        draw.text(
            (config.margin_px, y),
            line,
            fill=config.text_color,
            font=font,
        )
        y += line_height

    # Step 4: Split lines into pages, recording each page's line range
    num_pages = math.ceil(len(lines) / target_lpp)

    page_images: List[Image.Image] = []
    page_texts: List[str] = []
    page_line_ranges: List[Tuple[int, int]] = []  # (start_line, end_line) for each kept page

    for page_idx in range(num_pages):
        start_line = page_idx * target_lpp
        end_line = min(start_line + target_lpp, len(lines))

        # Trim leading blank lines
        while start_line < end_line and not lines[start_line].strip():
            start_line += 1
        # Trim trailing blank lines
        while end_line > start_line and not lines[end_line - 1].strip():
            end_line -= 1

        if start_line >= end_line:
            continue

        # Step 5: Build page image
        if config.tokens_per_page is not None:
            # Compression preset: render page lines on a fresh image to
            # avoid bleed from adjacent lines (tight line_height ≈ font_size).
            page_h = (end_line - start_line) * line_height + 2 * config.margin_px
            crop = Image.new("RGB", (config.width, page_h), config.bg_color)
            draw_page = ImageDraw.Draw(crop)
            py = config.margin_px
            for li in range(start_line, end_line):
                draw_page.text(
                    (config.margin_px, py), lines[li],
                    fill=config.text_color, font=font,
                )
                py += line_height
        else:
            # Corpus: crop from the shared tall image (original behavior)
            y_top = config.margin_px + start_line * line_height
            y_bot = config.margin_px + end_line * line_height
            crop = tall_img.crop((0, y_top, config.width, y_bot))

        # Step 6: Add page label
        crop = _add_page_label(crop, len(page_images) + 1)

        page_images.append(crop)
        page_texts.append("\n".join(lines[start_line:end_line]))
        page_line_ranges.append((start_line, end_line))

    if not page_images:
        return None

    # Step 7: Determine evidence pages by line-overlap with each page's range.
    # Deterministic; no string matching.
    evidence_pages: List[int] = []
    if evidence_line_ranges:
        ev_set: set = set()
        for first_line, last_line in evidence_line_ranges:
            for page_idx, (start_line, end_line) in enumerate(page_line_ranges):
                if first_line < end_line and last_line >= start_line:
                    ev_set.add(page_idx + 1)  # 1-indexed
        evidence_pages = sorted(ev_set)

    return PageRenderResult(
        pages=page_images,
        page_texts=page_texts,
        num_pages=len(page_images),
        config_key=config.to_key(),
        evidence_pages=evidence_pages,
    )


def render_from_page_texts(
    page_texts: List[str],
    config: RenderConfig,
) -> Optional[PageRenderResult]:
    """Render page images directly from pre-split page texts.

    Unlike render_pages(), this function skips text normalization and word
    wrapping. It assumes page_texts are already wrapped lines (as output by
    render_pages), and renders each page's lines directly into an image.

    This guarantees pixel-identical output when re-rendering from page_texts
    stored in evaluation data, regardless of the tokens_per_page setting.

    Args:
        page_texts: List of strings, one per page. Each string contains
            newline-separated lines (as produced by render_pages).
        config: RenderConfig controlling visual parameters.

    Returns:
        PageRenderResult with the rendered page images.
    """
    if not page_texts:
        return None

    font = ImageFont.truetype(config.font_path, config.font_size)
    line_height = config.line_height

    page_images: List[Image.Image] = []
    final_page_texts: List[str] = []

    for page_idx, page_text in enumerate(page_texts):
        lines = page_text.split("\n")

        # Trim blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        if not lines:
            continue

        # Render page image
        page_h = len(lines) * line_height + 2 * config.margin_px
        img = Image.new("RGB", (config.width, page_h), config.bg_color)
        draw = ImageDraw.Draw(img)
        y = config.margin_px
        for line in lines:
            draw.text((config.margin_px, y), line, fill=config.text_color, font=font)
            y += line_height

        # Add page label
        img = _add_page_label(img, len(page_images) + 1)

        page_images.append(img)
        final_page_texts.append("\n".join(lines))

    if not page_images:
        return None

    return PageRenderResult(
        pages=page_images,
        page_texts=final_page_texts,
        num_pages=len(page_images),
        config_key=config.to_key(),
    )

# ── Parallel rendering ─────────────────────────────────────────────────────

def _render_single_worker(args):
    """Worker function for parallel rendering. Picklable."""
    text, config_dict, evidence_spans, output_dir, sample_id = args
    from .rendering_config import RenderConfig

    config = RenderConfig(**config_dict)
    result = render_pages(text, config, evidence_spans=evidence_spans)
    if result is None:
        return None

    # Save images to disk if output_dir provided
    image_paths = []
    if output_dir:
        sample_dir = os.path.join(output_dir, config.to_key())
        os.makedirs(sample_dir, exist_ok=True)
        for i, page in enumerate(result.pages):
            path = os.path.join(sample_dir, f"{sample_id}_page_{i+1}.png")
            page.save(path)
            image_paths.append(path)

    return {
        "sample_id": sample_id,
        "num_pages": result.num_pages,
        "page_texts": result.page_texts,
        "image_paths": image_paths,
        "config_key": result.config_key,
        "evidence_pages": result.evidence_pages,
    }


def parallel_render_pages(
    samples: List[dict],
    config: "RenderConfig",
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    text_key: str = "context",
    id_key: str = "id",
    evidence_spans_key: Optional[str] = None,
) -> List[Optional[dict]]:
    """Render multiple samples in parallel using multiprocessing.

    Args:
        samples: List of sample dicts, each containing at minimum a text field.
        config: RenderConfig for rendering.
        output_dir: Directory to save rendered page images. If None, images
            are not saved (only page_texts and metadata returned).
        num_workers: Number of parallel workers. Defaults to CPU count.
        text_key: Key in sample dict for the text to render.
        id_key: Key in sample dict for the sample identifier.
        evidence_spans_key: Optional key for evidence spans (list of (start, end) tuples).

    Returns:
        List of result dicts (or None for failed renders), one per sample.
        Each dict contains: sample_id, num_pages, page_texts, image_paths,
        config_key, evidence_pages.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import cpu_count

    if num_workers is None:
        num_workers = min(cpu_count(), 32)

    # Serialize config to dict for pickling
    config_dict = {
        "width": config.width,
        "font_size": config.font_size,
        "line_spacing": config.line_spacing,
        "margin_px": config.margin_px,
        "font_path": config.font_path,
        "bg_color": config.bg_color,
        "text_color": config.text_color,
        "style_theme": config.style_theme,
        "tokens_per_page": config.tokens_per_page,
    }

    # Build task list
    tasks = []
    for s in samples:
        text = s.get(text_key, "")
        sample_id = s.get(id_key, f"sample_{len(tasks)}")
        evidence_spans = s.get(evidence_spans_key) if evidence_spans_key else None
        tasks.append((text, config_dict, evidence_spans, output_dir, sample_id))

    # Run in parallel
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(_render_single_worker, task): i
            for i, task in enumerate(tasks)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"  WARNING: Render failed for sample {idx}: {e}")
                results[idx] = None
            completed += 1
            if completed % 100 == 0:
                print(f"  Rendered {completed}/{len(tasks)} samples")

    successful = sum(1 for r in results if r is not None)
    print(f"  Parallel rendering complete: {successful}/{len(tasks)} successful")
    return results
