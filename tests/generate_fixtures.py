#!/usr/bin/env python3
"""Generate a small, reusable set of test images with real rendered text,
one per supported language plus a mixed-language page, for manually or
scriptedly checking that OCR extraction still works after a pipeline
change — no external downloads, no network dependency, no license
concerns (all fixture text below is original, written for this purpose).

Usage:
  python tests/generate_fixtures.py

Writes .png files to tests/fixtures/. Each is a plain, high-resolution
printed page — representative of the "clean text" case, not of the
harder layouts (tables, rotated sidebars, stylized covers) already
covered by the ad-hoc repros in extractor.py's own commit history. Use
these to sanity-check that ordinary per-language extraction hasn't
regressed; use a real document for anything layout-specific.
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Fonts confirmed available on macOS to render each script; swap these if
# running elsewhere (e.g. DejaVuSans.ttf + a Noto Armenian/Cyrillic build
# on Linux). Font choice measurably affects Armenian OCR accuracy even on
# otherwise-identical clean text — Mshtakan (a more traditional/stylized
# Armenian face) produced noticeably more dropped/misread words from
# Tesseract's "hye" model than Noto Sans Armenian's plainer letterforms
# did in a side-by-side test. Noto is used here as the fairer default;
# swap back to Mshtakan deliberately if you want to see the harder case.
FONT_LATIN = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_CYRILLIC = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_ARMENIAN = "/System/Library/Fonts/NotoSansArmenian.ttc"

# Short, original sentences (not excerpted from any copyrighted work) —
# each pack of 3 lines is a clean paragraph, easy to eyeball-check the
# extracted text against.
TEXT = {
    "english": (
        FONT_LATIN,
        [
            "The quick brown fox jumps over the lazy dog.",
            "Reading every day helps you learn new words and ideas.",
            "A good breakfast gives you energy for the whole morning.",
        ],
    ),
    "russian": (
        FONT_CYRILLIC,
        [
            "Быстрая лиса прыгает через ленивую собаку.",
            "Чтение каждый день помогает выучить новые слова.",
            "Хороший завтрак даёт энергию на всё утро.",
        ],
    ),
    "armenian": (
        FONT_ARMENIAN,
        [
            "Արագ աղվեսը ցատկում է ծույլ շան վրայով:",
            "Ամեն օր կարդալը օգնում է սովորել նոր բառեր:",
            "Լավ նախաճաշը էներգիա է տալիս ամբողջ առավոտվա համար:",
            "Գիրքը սեղանի վրա է, իսկ գրիչը՝ դարակում:",
            "Երեխաները խաղում են բակում արևոտ եղանակին:",
            "Ձմռանը լեռներում շատ ձյուն է լինում:",
        ],
    ),
}

# Default OCR_LANGS is "hye+rus+eng" combined — a page mixing all three
# scripts is the realistic default case, not an edge case, for this app.
MIXED_TEXT = [
    "The quick brown fox jumps over the lazy dog.",
    "Быстрая лиса прыгает через ленивую собаку.",
    "Արագ աղվեսը ցատկում է ծույլ շան վրայով:",
]


def render_page(lines_with_fonts, out_path: Path, width=1600, height=900, font_size=42):
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    y = 80
    for line, font_path in lines_with_fonts:
        font = ImageFont.truetype(font_path, font_size)
        d.text((80, y), line, fill="black", font=font)
        y += font_size + 30
    img.save(out_path)
    print(f"wrote {out_path}")


def main():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for lang, (font_path, lines) in TEXT.items():
        render_page([(line, font_path) for line in lines], FIXTURES_DIR / f"{lang}.png")

    mixed_fonts = [FONT_LATIN, FONT_CYRILLIC, FONT_ARMENIAN]
    render_page(list(zip(MIXED_TEXT, mixed_fonts)), FIXTURES_DIR / "mixed_hye_rus_eng.png")


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        print(f"Font load failed ({e}) — edit the FONT_* paths at the top of this "
              f"script for your OS.", file=sys.stderr)
        sys.exit(1)
