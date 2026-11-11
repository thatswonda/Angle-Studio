"""Thumbnail text: the suspense line is drawn on top of the picture, so it is always spelled right and reads at phone size.

Big ALL-CAPS words in a heavy font, white with a thick black outline and a soft drop shadow, one *highlighted* word
in bright yellow, on a darkened band so the words pop off any picture. The picture itself gets a little extra
colour and contrast, the way click-worthy thumbnails are graded.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .cards import FONTS, PIXEL, HAND, COMIC, _wrap

PLACES = ("top", "bottom", "left", "right")
HIGHLIGHT = (255, 214, 0)
WHITE = (255, 255, 255)
MAX_WORDS = 5


def _font_file(style_id: str) -> str:
    if style_id in PIXEL:
        return "PressStart2P.woff"
    if style_id in HAND:
        return "PermanentMarker.woff"
    if style_id in COMIC:
        return "Bangers.woff"
    return "Anton.woff"  # heavy and condensed: the classic thumbnail face, also on serif templates


def words_of(text: str) -> list[tuple[str, bool]]:
    """'it was *him*' -> [('IT', False), ('WAS', False), ('HIM', True)]; without a mark the last word is highlighted."""
    out: list[tuple[str, bool]] = []
    for tok in (text or "").replace("“", "").replace("”", "").replace('"', "").split():
        hi = "*" in tok
        w = tok.replace("*", "")
        if w:
            out.append((w.upper(), hi))
    out = out[:MAX_WORDS]
    if out and not any(h for _, h in out):
        w, _ = out[-1]
        out[-1] = (w, True)
    return out


def _place(place: str, w: int, h: int) -> str:
    place = place if place in PLACES else "top"
    if h > w and place in ("left", "right"):  # tall frames have no room beside the subject
        return "top"
    return place


def add_text(src: Path, text: str, place: str, style_id: str, out: Path) -> Path:
    img = Image.open(src).convert("RGB")
    img = ImageEnhance.Color(img).enhance(1.18)
    img = ImageEnhance.Contrast(img).enhance(1.08)
    w, h = img.size
    words = words_of(text)
    if not words:
        img.save(out, "JPEG", quality=92)
        return out
    place = _place(place, w, h)

    # The area the words sit in, and a dark fade behind it.
    if place in ("top", "bottom"):
        box_w, box_h = int(w * 0.9), int(h * (0.3 if h > w else 0.38))
        x0 = (w - box_w) // 2
        y0 = int(h * 0.04) if place == "top" else h - box_h - int(h * 0.05)
    else:
        box_w, box_h = int(w * 0.46), int(h * 0.82)
        x0 = int(w * 0.03) if place == "left" else w - box_w - int(w * 0.03)
        y0 = (h - box_h) // 2
    shade = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(shade)
    steps = 40
    for i in range(steps):
        a = int(150 * (1 - i / steps) ** 1.4)
        if place == "top":
            y = int((y0 + box_h * 1.25) * i / steps)
            sd.rectangle([0, y, w, y + h // steps + 2], fill=a)
        elif place == "bottom":
            y = h - int((h - y0 + box_h * 0.25) * i / steps)
            sd.rectangle([0, y - h // steps - 2, w, y], fill=a)
        elif place == "left":
            x = int((x0 + box_w * 1.2) * i / steps)
            sd.rectangle([x, 0, x + w // steps + 2, h], fill=a)
        else:
            x = w - int((w - x0 + box_w * 0.2) * i / steps)
            sd.rectangle([x - w // steps - 2, 0, x, h], fill=a)
    img = Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), img, shade.filter(ImageFilter.GaussianBlur(w // 60)))

    # Fit the words: as big as the area allows, at most 3 lines.
    draw = ImageDraw.Draw(img)
    ff = str(FONTS / _font_file(style_id))
    def fit(max_w):
        size = int(box_h * 0.6)
        while True:
            font = ImageFont.truetype(ff, size)
            lines = _wrap(words, font, max_w, draw)
            asc = font.getbbox("HG")
            line_h = asc[3] - asc[1]
            total = len(lines) * line_h + (len(lines) - 1) * line_h * 0.14
            widest = max(draw.textlength(" ".join(x for x, _ in ln), font=font) for ln in lines)
            if (len(lines) <= 3 and total <= box_h and widest <= max_w) or size <= 20:
                return size, font, lines, asc, line_h, total
            size = int(size * 0.93)

    # A short line on a narrow frame reads bigger when broken in two, so try both and keep the bigger words.
    size, font, lines, asc, line_h, total = max((fit(box_w), fit(int(box_w * 0.6))), key=lambda r: r[0])
    stroke = max(3, size // 11)
    gap = line_h * 0.14
    y = y0 + (box_h - total) / 2 - asc[1]
    space = draw.textlength(" ", font=font)

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld, shd = ImageDraw.Draw(layer), ImageDraw.Draw(shadow)
    for ln in lines:
        lw = draw.textlength(" ".join(x for x, _ in ln), font=font)
        if place == "left":
            x = x0
        elif place == "right":
            x = x0 + box_w - lw
        else:
            x = x0 + (box_w - lw) / 2
        for word, hi in ln:
            off = max(4, size // 14)
            shd.text((x + off, y + off), word, font=font, fill=(0, 0, 0, 200), stroke_width=stroke, stroke_fill=(0, 0, 0, 200))
            ld.text((x, y), word, font=font, fill=HIGHLIGHT if hi else WHITE, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += draw.textlength(word, font=font) + space
        y += line_h + gap
    img = img.convert("RGBA")
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(max(2, size // 18))))
    img.alpha_composite(layer)
    img.convert("RGB").save(out, "JPEG", quality=92)
    return out
