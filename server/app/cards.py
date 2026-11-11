"""Text cards: scenes drawn as designed words on a plain background, for free (no AI image model).

A card has text (a few words; *word* marks the word to highlight), an optional small line, and a layout:
big | number | question | quote. Colours come from the template's palette and the font from its kind of look.
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path(__file__).resolve().parent / "fonts"
MAX_WORDS = 3  # a card carries only the few words that matter most
# Colour emoji (Debian/Ubuntu package fonts-noto-color-emoji). Without it, cards are drawn without the emoji.
EMOJI_FONT = next((p for p in ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", "/usr/share/fonts/noto/NotoColorEmoji.ttf")
                   if Path(p).exists()), None)


def _emoji(e: str, size: int) -> Image.Image | None:
    """One colour emoji as an RGBA image about size px tall (the font only comes at 109 px, so it is scaled)."""
    if not EMOJI_FONT or not e or size < 8:
        return None
    try:
        f = ImageFont.truetype(EMOJI_FONT, 109)
        tmp = Image.new("RGBA", (220, 160), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((10, 10), e, font=f, embedded_color=True)
        box = tmp.getbbox()
        if not box:
            return None
        tmp = tmp.crop(box)
        k = size / tmp.height
        return tmp.resize((max(1, round(tmp.width * k)), size), Image.LANCZOS)
    except OSError:
        return None


def first_emoji(text: str) -> str:
    """The first emoji in a string (with its variation selector or skin tone), or ''."""
    m = re.search(r"(?:[\U0001F1E6-\U0001F1FF]{2}|[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\u2300-\u23FF]"
                  r"[\uFE0F\U0001F3FB-\U0001F3FF]?(?:\u200D[\U0001F000-\U0001FAFF\u2600-\u27BF][\uFE0F]?)*)", text or "")
    return m.group(0) if m else ""
LAYOUTS = ("big", "number", "question", "quote")

# Template -> display font. Anything not listed uses Anton (bold condensed).
HAND = {"stickman", "wiresketch", "lineart", "papercut"}
COMIC = {"comic", "cartoonexplain", "animated3d", "clay"}
SERIF = {"documentary", "historypaint", "watercolor", "botanical", "photoreal", "silhouette"}
PIXEL = {"pixel"}
# Templates whose cards look best on their dark colour.
DARK = {"cinematic", "retro", "silhouette", "toyworld", "anime"}


def _hex(c: str, default=(30, 30, 30)) -> tuple[int, int, int]:
    c = (c or "").lstrip("#")
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return default


def _lum(rgb) -> float:
    r, g, b = (x / 255 for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _font_for(style_id: str, kind: str) -> tuple[str, bool]:
    """(font file, upper-case the words?)"""
    if style_id in PIXEL:
        return "PressStart2P.woff", True
    if style_id in HAND:
        return "PermanentMarker.woff", True
    if style_id in COMIC or kind == "comic":
        return "Bangers.woff", True
    if style_id in SERIF or kind == "watercolor":
        return "PlayfairDisplay-ExtraBold.woff", False
    return "Anton.woff", True


def _colors(style: dict) -> tuple[tuple, tuple, tuple, tuple]:
    """background, text, accent, soft (for texture)"""
    pal = [_hex(c) for c in (style.get("palette") or ["#1E2A33", "#5E7A8A", "#F4EEE2", "#E4572E"])][:4]
    while len(pal) < 4:
        pal.append((200, 200, 200))
    dark, mid, light, accent = pal
    if style.get("id") in DARK:
        bg, fg = dark, light
    else:
        bg, fg = light, dark
    if abs(_lum(bg) - _lum(fg)) < 0.45:  # make sure the words always read clearly
        fg = (20, 20, 20) if _lum(bg) > 0.5 else (248, 246, 240)
    if abs(_lum(accent) - _lum(bg)) < 0.18:
        accent = mid if abs(_lum(mid) - _lum(bg)) >= 0.18 else fg
    return bg, fg, accent, mid


def _words(text: str) -> list[tuple[str, bool]]:
    """'save *first*' -> [('save', False), ('first', True)]"""
    out = []
    for tok in re.findall(r"\*[^*]+\*|\S+", text.strip()):
        if tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            out.extend((w, True) for w in tok[1:-1].split())
        else:
            out.append((tok, False))
    if len(out) <= MAX_WORDS:
        return out
    # too many words: keep the highlighted word, drop small filler words, then keep the words nearest the highlight
    filler = {"a", "an", "the", "of", "to", "and", "or", "is", "was", "are", "were", "in", "on", "at", "for", "with", "his", "her",
              "their", "its", "my", "your", "our", "this", "that", "it", "be", "by", "so", "very", "just"}
    idx = [i for i, (w, hi) in enumerate(out) if not (w.lower().strip(".,!?") in filler and not hi)]
    if len(idx) <= MAX_WORDS:
        return [out[i] for i in idx]
    focus = next((i for i in idx if out[i][1]), None)
    if focus is None:  # nothing highlighted: the first meaningful words carry the subject
        return [out[i] for i in idx[:MAX_WORDS]]
    keep = sorted(sorted(idx, key=lambda i: (abs(i - focus), i > focus))[:MAX_WORDS])
    return [out[i] for i in keep]


def _wrap(words, font, max_w, draw):
    lines, cur = [], []
    for w in words:
        trial = cur + [w]
        if cur and draw.textlength(" ".join(x for x, _ in trial), font=font) > max_w:
            lines.append(cur)
            cur = [w]
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _fit(words, font_file, max_w, max_h, draw, start, min_size=24, max_lines=3):
    size = start
    while size >= min_size:
        font = ImageFont.truetype(str(FONTS / font_file), size)
        lines = _wrap(words, font, max_w, draw)
        line_h = font.getbbox("Hg")[3] - font.getbbox("Hg")[1]
        total = len(lines) * line_h * 1.12
        widest = max(draw.textlength(" ".join(x for x, _ in ln), font=font) for ln in lines)
        if len(lines) <= max_lines and total <= max_h and widest <= max_w:
            return font, lines, line_h
        size = int(size * 0.92)
    font = ImageFont.truetype(str(FONTS / font_file), min_size)
    lines = _wrap(words, font, max_w, draw)
    return font, lines, font.getbbox("Hg")[3] - font.getbbox("Hg")[1]


def _background(w, h, bg, soft, style_id, seed) -> Image.Image:
    rnd = random.Random(seed)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    if style_id == "silhouette":  # glowing screen, amber to deep red
        top, bottom = _hex("#F6B256"), _hex("#7A2E1E")
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3)))
    elif style_id in COMIC or style_id == "comic":  # halftone dots in two corners
        r = max(3, w // 160)
        for cx, cy in ((0, 0), (w, h)):
            for y in range(0, h // 3, r * 4):
                for x in range(0, w // 2, r * 4):
                    px, py = (x, y) if cx == 0 else (w - x, h - y)
                    k = 1 - math.hypot(x / (w / 2), y / (h / 3))
                    if k > 0:
                        rr = r * k * 1.4
                        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=soft)
    # soft vignette
    v = Image.new("L", (w, h), 0)
    vd = ImageDraw.Draw(v)
    vd.ellipse([-w * 0.25, -h * 0.2, w * 1.25, h * 1.2], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(min(w, h) // 6))
    shade = Image.new("RGB", (w, h), tuple(max(0, int(c * 0.86)) for c in img.getpixel((w // 2, h // 2))))
    img = Image.composite(img, shade, v)
    # fine paper grain
    if style_id not in PIXEL:
        g = Image.effect_noise((w // 2, h // 2), 18).resize((w, h)).convert("L")
        img = Image.blend(img, Image.merge("RGB", (g, g, g)), 0.035)
    return img


def make_card(card: dict, style: dict, w: int, h: int, out: Path) -> Path:
    """Draws one text card at w x h and saves it as PNG."""
    style_id, kind = style.get("id", ""), style.get("kind", "")
    layout = card.get("layout") if card.get("layout") in LAYOUTS else "big"
    font_file, upper = _font_for(style_id, kind)
    bg, fg, accent, soft = _colors(style)
    emoji = first_emoji(str(card.get("emoji") or ""))
    text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]", "", str(card.get("text") or "")).strip() or "..."
    if upper:
        text = text.upper()
    small = str(card.get("small") or "").strip()
    seed = sum(map(ord, text)) + len(small)
    pixel = style_id in PIXEL
    W, H = (w // 4, h // 4) if pixel else (w, h)  # pixel cards are drawn small and scaled up with hard edges
    img = _background(W, H, bg, soft, style_id, seed)
    d = ImageDraw.Draw(img)
    short = min(W, H)
    box_w = W * 0.78
    words = _words(text)
    top_space = 0

    if layout == "number":  # big accent numeral, words under it
        num = re.search(r"\d+", small or text)
        numeral = num.group(0) if num else "1"
        nf = ImageFont.truetype(str(FONTS / font_file), int(short * 0.42))
        nb = d.textbbox((0, 0), numeral, font=nf)
        nw, nh = nb[2] - nb[0], nb[3] - nb[1]
        top_space = nh + short * 0.06
        words = [x for x in words if x[0] != numeral and x[0] != "#" + numeral] or words
    em = _emoji(emoji, int(short * (0.2 if layout == "number" else 0.3)))
    em_h = em.height + short * 0.07 if em else 0
    font, lines, line_h = _fit(words, font_file, box_w, H * 0.36, d, int(short * (0.22 if layout != "quote" else 0.15)),
                               min_size=max(10, int(short * 0.05)), max_lines=2)
    gap = line_h * 1.12
    block_h = len(lines) * gap
    small_font = ImageFont.truetype(str(FONTS / ("PressStart2P.woff" if pixel else "Inter-SemiBold.woff")), max(9, int(short * 0.05)))
    small_h = (small_font.getbbox("Hg")[3] + short * 0.04) if small and layout != "number" else 0
    y = (H - (top_space + em_h + block_h + small_h)) / 2

    if layout == "number":
        nb = d.textbbox((0, 0), numeral, font=nf)
        d.text(((W - (nb[2] - nb[0])) / 2 - nb[0], y - nb[1]), numeral, font=nf, fill=accent)
        y += top_space
    elif layout == "question":  # a big faint question mark behind the words
        qf = ImageFont.truetype(str(FONTS / font_file), int(short * 0.9))
        qb = d.textbbox((0, 0), "?", font=qf)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).text(((W - (qb[2] - qb[0])) / 2 - qb[0], (H - (qb[3] - qb[1])) / 2 - qb[1]), "?", font=qf, fill=accent + (70,))
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
        d = ImageDraw.Draw(img)
    elif layout == "quote":  # a large accent quotation mark above
        qf = ImageFont.truetype(str(FONTS / "PlayfairDisplay-ExtraBold.woff"), int(short * 0.32))
        qb = d.textbbox((0, 0), "“", font=qf)
        d.text(((W - (qb[2] - qb[0])) / 2 - qb[0], y - (qb[3] - qb[1]) * 0.95 - qb[1]), "“", font=qf, fill=accent)

    if em:  # the emoji sits above the words, with a soft shadow
        ex = int((W - em.width) / 2)
        sh = Image.new("RGBA", em.size, (0, 0, 0, 0))
        sh.putalpha(em.getchannel("A").point(lambda a: a * 0.25))
        sh = sh.filter(ImageFilter.GaussianBlur(max(1, short // 120)))
        base = img.convert("RGBA")
        base.alpha_composite(sh, (ex, int(y + short * 0.015)))
        base.alpha_composite(em, (ex, int(y)))
        img = base.convert("RGB")
        d = ImageDraw.Draw(img)
        y += em_h
    shadow = (0, 0, 0) if _lum(bg) < 0.5 else None
    top = font.getbbox("Hg")[1]  # the font's own top gap: y is where the letters visibly start
    for ln in lines:
        widths = [d.textlength(wd, font=font) for wd, _ in ln]
        space = d.textlength(" ", font=font)
        x = (W - (sum(widths) + space * (len(ln) - 1))) / 2
        for (wd, hi), ww in zip(ln, widths):
            if shadow is not None:
                d.text((x + short * 0.006, y - top + short * 0.006), wd, font=font, fill=shadow)
            d.text((x, y - top), wd, font=font, fill=accent if hi else fg)
            x += ww + space
        y += gap
    if layout == "big":  # an accent bar under the words
        bar_w = min(box_w * 0.5, short * 0.32)
        bh = max(2, int(short * 0.014))
        by = y - gap + line_h + short * 0.045
        if style_id in HAND:  # a hand-drawn underline
            pts = [((W - bar_w) / 2 + bar_w * t / 12, by + math.sin(t * 1.3 + seed) * bh * 0.6) for t in range(13)]
            d.line(pts, fill=accent, width=bh, joint="curve")
        else:
            d.rounded_rectangle([(W - bar_w) / 2, by, (W + bar_w) / 2, by + bh], radius=bh // 2, fill=accent)
        y = by + bh + short * 0.03
    if small and layout != "number":
        sb = d.textbbox((0, 0), small, font=small_font)
        y_small = y if layout == "big" else y - gap + line_h + short * 0.05
        d.text(((W - (sb[2] - sb[0])) / 2 - sb[0], y_small - sb[1]), small, font=small_font, fill=soft if abs(_lum(soft) - _lum(bg)) > 0.25 else fg)
    if pixel:
        img = img.resize((w, h), Image.NEAREST)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out
