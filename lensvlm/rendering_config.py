# Copyright (C) 2026 Apple Inc. All Rights Reserved.
"""
Rendering configuration.
"""

import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── Font discovery ─────────────────────────────────────────────────────────

_CANDIDATE_SERIF = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed-Bold.ttf",
]

_CANDIDATE_SANS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Light.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Medium.ttf",
    "/usr/share/fonts/truetype/lato/Lato-Semibold.ttf",
]

_CANDIDATE_MONO = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
]


def _filter_existing(paths: List[str]) -> List[str]:
    return [p for p in paths if os.path.exists(p)]


def _discover_fonts() -> tuple:
    """Find available fonts, falling back to bundled DejaVuSans if needed."""
    serif = _filter_existing(_CANDIDATE_SERIF)
    sans = _filter_existing(_CANDIDATE_SANS)
    mono = _filter_existing(_CANDIDATE_MONO)

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bundled = os.path.join(_project_root, "fonts", "DejaVuSans.ttf")
    if os.path.exists(bundled) and not sans:
        sans = [bundled]
    if not (serif or sans or mono):
        for root, _, files in os.walk("/usr/share/fonts"):
            for f in files:
                if f.endswith(".ttf"):
                    sans.append(os.path.join(root, f))
                    if len(sans) >= 5:
                        break
            if sans:
                break
    if not (serif or sans or mono):
        if os.path.exists(bundled):
            sans = [bundled]
        else:
            raise RuntimeError("No TTF fonts found on system or in fonts/ dir")

    all_fonts = serif + sans + mono
    return serif, sans, mono, all_fonts


FONTS_SERIF, FONTS_SANS, FONTS_MONO, FONTS_ALL = _discover_fonts()

# ── Color palettes ─────────────────────────────────────────────────────────
# (background, text_color)

LIGHT_PALETTES = [
    ((255, 255, 255), (0, 0, 0)),
    ((255, 255, 255), (33, 33, 33)),
    ((250, 250, 245), (40, 40, 40)),
    ((245, 245, 240), (50, 50, 50)),
    ((240, 240, 235), (30, 30, 30)),
    ((248, 248, 255), (25, 25, 50)),
    ((255, 253, 245), (60, 50, 30)),
]

DARK_PALETTES = [
    ((30, 30, 30), (220, 220, 220)),
    ((20, 20, 25), (200, 200, 210)),
    ((35, 35, 40), (230, 230, 225)),
    ((25, 25, 30), (180, 200, 180)),
    ((40, 35, 30), (210, 200, 180)),
    ((20, 30, 40), (200, 210, 220)),
]

WEB_PALETTES = [
    ((255, 255, 255), (51, 51, 51)),
    ((250, 250, 250), (34, 34, 34)),
    ((245, 245, 245), (68, 68, 68)),
    ((255, 252, 240), (60, 50, 40)),
    ((240, 248, 255), (30, 30, 60)),
]

ARTISTIC_PALETTES = [
    ((255, 255, 240), (70, 60, 50)),
    ((230, 220, 200), (50, 40, 20)),
    ((210, 200, 180), (40, 30, 10)),
    ((245, 240, 230), (80, 60, 40)),
    ((200, 210, 200), (30, 40, 30)),
    ((220, 220, 230), (40, 40, 60)),
]

# Default font path (bundled)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_FONT = os.path.join(_PROJECT_ROOT, "fonts", "DejaVuSans.ttf")

# Target tokens per page (500-1000, midpoint 750)
TARGET_TOKENS_PER_PAGE = 750
CHARS_PER_TOKEN = 4  # ~4 chars per token

# Compression rate presets for QA data.
# compression_rate = tokens_per_page / visual_tokens_per_page
# All use margin_px=8, font >= 7, <= 500 tokens/page.
COMPRESSION_PRESETS = {
    "5x":  {"width": 256, "font_size": 8, "line_spacing": 1.15, "margin_px": 7,
             "tokens_per_page": 405},
    "10x": {"width": 192, "font_size": 6, "line_spacing": 1.1,  "margin_px": 6,
             "tokens_per_page": 540},
    "15x": {"width": 128, "font_size": 5, "line_spacing": 1.05, "margin_px": 5,
             "tokens_per_page": 378},
}


@dataclass
class RenderConfig:
    """Rendering configuration for a single sample.

    Width-based render config (pixel units).
    """

    # Layout
    width: int = 700
    font_size: int = 10
    line_spacing: float = 1.2
    margin_px: int = 8

    # Typography
    font_path: str = DEFAULT_FONT

    # Colors
    bg_color: Tuple[int, int, int] = (255, 255, 255)
    text_color: Tuple[int, int, int] = (0, 0, 0)

    # Theme name (for tracking)
    style_theme: str = "document"

    # Tokens per page override (set by compression presets, None = use default)
    tokens_per_page: Optional[int] = None

    # ── Derived values ────────────────────────────────────────────────

    @property
    def line_height(self) -> int:
        return max(1, int(self.font_size * self.line_spacing))

    @property
    def text_area_width(self) -> int:
        return max(1, self.width - 2 * self.margin_px)

    def chars_per_line_estimate(self) -> int:
        char_width = self.font_size * 0.55
        return max(1, int(self.text_area_width / char_width))

    def target_lines_per_page(self) -> int:
        """Compute lines per page to hit target tokens per page."""
        chars_per_line = self.chars_per_line_estimate()
        tpp = self.tokens_per_page if self.tokens_per_page is not None else TARGET_TOKENS_PER_PAGE
        target_chars = tpp * CHARS_PER_TOKEN
        return max(5, target_chars // chars_per_line)

    def visual_tokens_for_page(self, page_height: int) -> int:
        """Qwen visual tokens: ceil(h/32) * ceil(w/32)."""
        return math.ceil(max(1, page_height) / 32) * math.ceil(self.width / 32)

    def avg_visual_tokens_per_page(self) -> int:
        """Estimate visual tokens for an average content-tight page."""
        avg_height = self.margin_px * 2 + self.target_lines_per_page() * self.line_height
        return self.visual_tokens_for_page(avg_height)

    def max_pages_for_budget(self, max_total_tokens: int, text_chars: int,
                             image_max_pixels: int = 0) -> int:
        """Max pages that fit within a token budget."""
        vt_per_page = self.avg_visual_tokens_per_page()
        text_tokens = text_chars / CHARS_PER_TOKEN
        available = max_total_tokens - text_tokens
        if available <= 0:
            return 1
        return max(1, int(available / vt_per_page))

    def to_key(self) -> str:
        return f"{self.style_theme}_w{self.width}_f{self.font_size}"


# ── Style theme samplers ───────────────────────────────────────────────────

def _sample_width(theme: str) -> int:
    """Sample image width in pixels. Range: 400-800, centered on 700."""
    if theme == "artistic_pixel":
        return random.choices(
            [random.randint(400, 499), random.randint(500, 599), random.randint(600, 700)],
            weights=[0.3, 0.5, 0.2],
        )[0]
    elif theme == "web":
        return random.choices(
            [random.randint(600, 699), random.randint(700, 749), random.randint(750, 800)],
            weights=[0.2, 0.5, 0.3],
        )[0]
    else:
        # document / dark_mode: balanced around 600-700
        return random.choices(
            [random.randint(400, 549), random.randint(550, 699), random.randint(700, 800)],
            weights=[0.15, 0.5, 0.35],
        )[0]


def _sample_font(theme: str) -> str:
    if theme == "document":
        pool = FONTS_SERIF + FONTS_SANS[:4]
    elif theme == "web":
        pool = FONTS_SANS if FONTS_SANS else FONTS_ALL
    elif theme == "dark_mode":
        pool = FONTS_SANS + FONTS_MONO
    elif theme == "artistic_pixel":
        pool = FONTS_MONO + FONTS_SERIF[:2]
    else:
        pool = FONTS_ALL
    if not pool:
        pool = FONTS_ALL
    return random.choice(pool)


def _sample_font_size(theme: str) -> int:
    """Sample font size in pixels. Range: 7-12."""
    sizes = [7, 8, 9, 10, 11, 12]
    if theme == "artistic_pixel":
        weights = [0.25, 0.30, 0.20, 0.15, 0.05, 0.05]
    elif theme == "web":
        weights = [0.05, 0.05, 0.15, 0.30, 0.25, 0.20]
    else:
        # document / dark_mode: centered on 9-10
        weights = [0.05, 0.10, 0.20, 0.30, 0.20, 0.15]
    return random.choices(sizes, weights=weights)[0]


def _sample_line_spacing(theme: str) -> float:
    """Sample line spacing multiplier. Range: 1.1-1.4, centered on 1.2."""
    options = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    if theme == "artistic_pixel":
        weights = [0.10, 0.15, 0.25, 0.20, 0.15, 0.10, 0.05]
    elif theme == "web":
        weights = [0.05, 0.10, 0.20, 0.25, 0.20, 0.15, 0.05]
    else:
        weights = [0.05, 0.15, 0.30, 0.25, 0.15, 0.05, 0.05]
    return random.choices(options, weights=weights)[0]


def _sample_margin(theme: str) -> int:
    """Sample margin in pixels. Range: 4-16, centered on 8."""
    if theme == "artistic_pixel":
        return random.randint(4, 8)
    elif theme == "web":
        return random.randint(6, 14)
    else:
        return random.randint(6, 12)


def _sample_colors(theme: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if theme == "dark_mode":
        return random.choice(DARK_PALETTES)
    elif theme == "web":
        return random.choice(WEB_PALETTES)
    elif theme == "artistic_pixel":
        return random.choice(ARTISTIC_PALETTES)
    else:
        return random.choice(LIGHT_PALETTES)


def sample_render_config(
    theme: Optional[str] = None,
    compression: Optional[str] = None,
    tokens_per_page_override: Optional[int] = None,
) -> RenderConfig:
    """Build a fixed RenderConfig for the given compression preset.

    Args:
        theme: Ignored (kept for callsite compatibility). Always "document".
        compression: Required. One of "5x", "10x", "15x", or "random".
                     If None, defaults to "random" so legacy callers still work.
        tokens_per_page_override: Optional override for the preset's
                                  tokens_per_page.
    """
    if compression is None or compression == "random":
        compression = random.choice(list(COMPRESSION_PRESETS.keys()))
    if compression not in COMPRESSION_PRESETS:
        raise ValueError(
            f"Unknown compression preset: {compression!r}. "
            f"Choose from {list(COMPRESSION_PRESETS.keys())} or 'random'"
        )

    preset = COMPRESSION_PRESETS[compression]
    tpp = tokens_per_page_override if tokens_per_page_override else preset["tokens_per_page"]
    return RenderConfig(
        width=preset["width"],
        font_size=preset["font_size"],
        line_spacing=preset["line_spacing"],
        margin_px=preset["margin_px"],
        font_path=DEFAULT_FONT,            # locked: DejaVuSans.ttf
        bg_color=(255, 255, 255),          # locked: white
        text_color=(0, 0, 0),              # locked: black
        style_theme="document",            # locked
        tokens_per_page=tpp,
    )
