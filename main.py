# save as glyph_to_svg.py
import freetype as ft
from pathlib import Path

# Map readable targets to FreeType load target flags
_LOAD_TARGET = {
    "normal": ft.FT_LOAD_TARGET_NORMAL,
    "light":  ft.FT_LOAD_TARGET_LIGHT,
    "mono":   ft.FT_LOAD_TARGET_MONO,
    "lcd":    ft.FT_LOAD_TARGET_LCD,
    "lcd_v":  ft.FT_LOAD_TARGET_LCD_V,
}

def glyph_to_svg(
    font_path: str,
    char: str,
    ppem: int = 64,
    dpi: int = 72,
    hinting_target: str = "normal",
    out_svg: str = "glyph.svg",
):
    """
    Writes an SVG showing the hinted outline of `char` in `font_path`,
    plus metrics (baseline, asc/desc, origin, LSB, advance, glyph bbox).
    """

    face = ft.Face(font_path)
    face.set_char_size(0, ppem * 64, dpi, dpi)

    # Load hinted, scalable outline (avoid embedded bitmaps).
    flags = ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
    face.load_char(char, flags)

    slot = face.glyph
    outline = slot.outline
    m = slot.metrics  # FT_Glyph_Metrics, in 26.6 pixels at this size
    scale = 1.0 / 64.0

    def px(v):  # 26.6 -> float pixels
        return v * scale

    # --- Decompose the outline to an SVG path -------------------------------
    path_cmds = []
    contour_open = False

    def move_to(p, _):
        nonlocal contour_open
        if contour_open:
            path_cmds.append("Z")
        path_cmds.append(f"M {px(p.x):.3f} {-px(p.y):.3f}")
        contour_open = True

    def line_to(p, _):
        path_cmds.append(f"L {px(p.x):.3f} {-px(p.y):.3f}")

    def conic_to(p1, p2, _):
        # Quadratic Bézier in SVG: Q cx cy x y
        path_cmds.append(
            f"Q {px(p1.x):.3f} {-px(p1.y):.3f} {px(p2.x):.3f} {-px(p2.y):.3f}"
        )

    def cubic_to(p1, p2, p3, _):
        # Cubic Bézier in SVG: C c1x c1y c2x c2y x y
        path_cmds.append(
            "C "
            f"{px(p1.x):.3f} {-px(p1.y):.3f} "
            f"{px(p2.x):.3f} {-px(p2.y):.3f} "
            f"{px(p3.x):.3f} {-px(p3.y):.3f}"
        )

    outline.decompose(move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
    if contour_open:
        path_cmds.append("Z")

    d_attr = " ".join(path_cmds)

    # --- Metrics (all in pixel space, y-up; we flip sign for SVG y-down) ----
    lsb   = px(m.horiBearingX)     # left side bearing: x of glyph bbox left edge
    top   = px(m.horiBearingY)     # distance from baseline up to glyph bbox top
    bbox_w = px(m.width)
    bbox_h = px(m.height)
    adv   = px(m.horiAdvance)

    asc   = px(face.size.ascender)
    desc  = px(face.size.descender)  # negative

    # --- ViewBox: include metrics guides with a margin -----------------------
    margin = max(ppem * 0.2, 5)  # pixels
    left   = min(0.0, lsb) - margin
    right  = max(adv, lsb + bbox_w) + margin
    top_svg    = -(asc + margin)           # SVG y is down
    height_svg = (asc - desc) + 2 * margin
    width_svg  = right - left

    # Convenience for labeling
    def label(x, y, text, anchor="start"):
        return f'<text x="{x:.3f}" y="{y:.3f}" font-size="10" text-anchor="{anchor}" ' \
               f'font-family="monospace">{text}</text>'

    # --- Build the SVG -------------------------------------------------------
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{left:.3f} {top_svg:.3f} {width_svg:.3f} {height_svg:.3f}" '
        f'width="{width_svg:.0f}" height="{height_svg:.0f}">'
    )

    # Background (optional, light)
    svg.append('<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="#f9f9fb"/>'
               .format(left, top_svg, width_svg, height_svg))

    # Baseline, ascender, descender
    svg.append(f'<line x1="{left:.3f}" y1="{0:.3f}" x2="{right:.3f}" y2="{0:.3f}" stroke="#c0c0c0" stroke-dasharray="4 3"/>')
    svg.append(f'<line x1="{left:.3f}" y1="{-asc:.3f}" x2="{right:.3f}" y2="{-asc:.3f}" stroke="#e0a000" stroke-dasharray="4 3"/>')
    svg.append(f'<line x1="{left:.3f}" y1="{-desc:.3f}" x2="{right:.3f}" y2="{-desc:.3f}" stroke="#e0a000" stroke-dasharray="4 3"/>')
    svg.append(label(left + 3, -asc - 3, "ascender"))
    svg.append(label(left + 3,  -3,       "baseline"))
    svg.append(label(left + 3, -desc + 12, "descender"))

    # Origin and LSB line
    svg.append(f'<line x1="{0:.3f}" y1="{-asc:.3f}" x2="{0:.3f}" y2="{-desc:.3f}" stroke="#c0c0c0" stroke-dasharray="4 3"/>')
    svg.append(label(0, -asc - 6, "origin x=0", anchor="middle"))
    svg.append(f'<line x1="{lsb:.3f}" y1="{-asc:.3f}" x2="{lsb:.3f}" y2="{-desc:.3f}" stroke="#00a0e0" stroke-dasharray="4 3"/>')
    svg.append(label(lsb, -asc - 6, f"LSB={lsb:.2f}px", anchor="middle"))

    # Advance width line
    svg.append(f'<line x1="{adv:.3f}" y1="{-asc:.3f}" x2="{adv:.3f}" y2="{-desc:.3f}" stroke="#008000" stroke-dasharray="4 3"/>')
    svg.append(label(adv, -asc - 6, f"advance={adv:.2f}px", anchor="middle"))

    # Glyph bbox rectangle (from metrics)
    svg.append(
        f'<rect x="{lsb:.3f}" y="{-top:.3f}" width="{bbox_w:.3f}" height="{bbox_h:.3f}" '
        f'fill="none" stroke="#cc3333" stroke-width="1"/>'
    )
    svg.append(label(lsb + 3, -top + 12, f"bbox {bbox_w:.2f}×{bbox_h:.2f}px"))

    # The glyph outline
    svg.append(f'<path d="{d_attr}" fill="none" stroke="#000" stroke-width="1.2"/>')

    svg.append("</svg>")

    Path(out_svg).write_text("\n".join(svg), encoding="utf-8")
    return {
        "char": char,
        "ppem": ppem,
        "dpi": dpi,
        "lsb": lsb,
        "advance": adv,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "ascender": asc,
        "descender": desc,
        "out_svg": str(Path(out_svg).resolve()),
    }


if __name__ == "__main__":
    # Example usage
    info = glyph_to_svg("ARIALUNI.TTF", "a", ppem=12, dpi=96, hinting_target="mono", out_svg="a_hint.svg")
    print("Wrote:", info["out_svg"])
    print({k: v for k, v in info.items() if k != "out_svg"})
