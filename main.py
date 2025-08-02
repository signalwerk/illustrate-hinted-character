# save as glyph_to_svg_compare.py
import freetype as ft
from pathlib import Path

_LOAD_TARGET = {
    "normal": ft.FT_LOAD_TARGET_NORMAL,
    "light":  ft.FT_LOAD_TARGET_LIGHT,
    "mono":   ft.FT_LOAD_TARGET_MONO,
    "lcd":    ft.FT_LOAD_TARGET_LCD,
    "lcd_v":  ft.FT_LOAD_TARGET_LCD_V,
}

def glyph_to_svg_compare(
    font_path: str,
    char: str,
    ppem: int = 64,
    dpi: int = 72,
    hinting_target: str = "normal",
    out_svg: str = "glyph_compare.svg",
):
    """
    Writes an SVG that overlays:
      • Hinted outline (grid-fitted) — black
      • Original outline (same size, unhinted) — purple dashed
    Plus metrics (baseline, ascender, descender, origin, LSB, advance, bbox).
    """

    face = ft.Face(font_path)
    face.set_char_size(0, ppem * 64, dpi, dpi)
    scale = 1.0 / 64.0

    def px(v):  # 26.6 -> float pixels
        return v * scale

    def outline_to_path(outline):
        """Decompose an FT_Outline into an SVG path string (y flipped for SVG)."""
        cmds = []
        contour_open = False

        def move_to(p, _):
            nonlocal contour_open
            if contour_open:
                cmds.append("Z")
            cmds.append(f"M {px(p.x):.3f} {-px(p.y):.3f}")
            contour_open = True

        def line_to(p, _):
            cmds.append(f"L {px(p.x):.3f} {-px(p.y):.3f}")

        def conic_to(c, p, _):
            cmds.append(f"Q {px(c.x):.3f} {-px(c.y):.3f} {px(p.x):.3f} {-px(p.y):.3f}")

        def cubic_to(c1, c2, p, _):
            cmds.append(
                "C "
                f"{px(c1.x):.3f} {-px(c1.y):.3f} "
                f"{px(c2.x):.3f} {-px(c2.y):.3f} "
                f"{px(p.x):.3f} {-px(p.y):.3f}"
            )

        outline.decompose(move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
        if contour_open:
            cmds.append("Z")
        return " ".join(cmds)

    # --- Load hinted outline (grid-fitted) ---
    flags_hinted = ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
    face.load_char(char, flags_hinted)
    hinted_slot = face.glyph
    hinted_path = outline_to_path(hinted_slot.outline)
    hinted_metrics = hinted_slot.metrics

    # --- Load original (unhinted) outline at the same size ---
    flags_unhinted = ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | ft.FT_LOAD_NO_HINTING
    face.load_char(char, flags_unhinted)
    orig_slot = face.glyph
    orig_path = outline_to_path(orig_slot.outline)
    orig_metrics = orig_slot.metrics  # not used for guides (we keep guides from hinted)

    # Metrics for guides (use hinted to match what will actually render)
    lsb   = px(hinted_metrics.horiBearingX)
    top   = px(hinted_metrics.horiBearingY)
    bbox_w = px(hinted_metrics.width)
    bbox_h = px(hinted_metrics.height)
    adv   = px(hinted_metrics.horiAdvance)

    # Correct asc/desc (thanks!)
    asc   = px(face.size.ascender)
    desc  = px(face.size.descender)  # negative

    # ViewBox to include all guides with a margin
    margin = max(ppem * 0.2, 5)
    left   = min(0.0, lsb) - margin
    right  = max(adv, lsb + bbox_w) + margin
    top_svg    = -(asc + margin)
    height_svg = (asc - desc) + 2 * margin
    width_svg  = right - left

    def label(x, y, text, anchor="start"):
        return f'<text x="{x:.3f}" y="{y:.3f}" font-size="10" text-anchor="{anchor}" font-family="monospace">{text}</text>'

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{left:.3f} {top_svg:.3f} {width_svg:.3f} {height_svg:.3f}" '
        f'width="{width_svg:.0f}" height="{height_svg:.0f}">'
    )

    # Background
    svg.append('<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="#f9f9fb"/>'
               .format(left, top_svg, width_svg, height_svg))

    # Baseline, ascender, descender
    svg.append(f'<line x1="{left:.3f}" y1="{0:.3f}" x2="{right:.3f}" y2="{0:.3f}" stroke="#c0c0c0" stroke-dasharray="4 3"/>')
    svg.append(f'<line x1="{left:.3f}" y1="{-asc:.3f}" x2="{right:.3f}" y2="{-asc:.3f}" stroke="#e0a000" stroke-dasharray="4 3"/>')
    svg.append(f'<line x1="{left:.3f}" y1="{-desc:.3f}" x2="{right:.3f}" y2="{-desc:.3f}" stroke="#e0a000" stroke-dasharray="4 3"/>')
    svg.append(label(left + 3, -asc - 3, "ascender"))
    svg.append(label(left + 3, -3, "baseline"))
    svg.append(label(left + 3, -desc + 12, "descender"))

    # Origin, LSB, advance (from hinted metrics)
    svg.append(f'<line x1="0" y1="{-asc:.3f}" x2="0" y2="{-desc:.3f}" stroke="#c0c0c0" stroke-dasharray="4 3"/>')
    svg.append(label(0, -asc - 6, "origin x=0", anchor="middle"))
    svg.append(f'<line x1="{lsb:.3f}" y1="{-asc:.3f}" x2="{lsb:.3f}" y2="{-desc:.3f}" stroke="#00a0e0" stroke-dasharray="4 3"/>')
    svg.append(label(lsb, -asc - 6, f"LSB={lsb:.2f}px", anchor="middle"))
    svg.append(f'<line x1="{adv:.3f}" y1="{-asc:.3f}" x2="{adv:.3f}" y2="{-desc:.3f}" stroke="#008000" stroke-dasharray="4 3"/>')
    svg.append(label(adv, -asc - 6, f"advance={adv:.2f}px", anchor="middle"))

    # Glyph bbox (hinted metrics)
    svg.append(
        f'<rect x="{lsb:.3f}" y="{-top:.3f}" width="{bbox_w:.3f}" height="{bbox_h:.3f}" '
        f'fill="none" stroke="#cc3333" stroke-width="1"/>'
    )
    svg.append(label(lsb + 3, -top + 12, f"bbox {bbox_w:.2f}×{bbox_h:.2f}px"))

    # --- Outlines ---
    # Original (unhinted) outline — dashed purple
    svg.append(f'<path d="{orig_path}" fill="none" stroke="#7a3fe0" stroke-width="1.2" stroke-dasharray="5 3" opacity="0.9"/>')
    # Hinted outline — solid black on top
    svg.append(f'<path d="{hinted_path}" fill="none" stroke="#000000" stroke-width="1.2"/>')

    # Legend
    legend_x = left + 8
    legend_y = -asc + 16
    svg.append(label(legend_x + 24, legend_y, "original (unhinted)", anchor="start"))
    svg.append(f'<line x1="{legend_x:.3f}" y1="{legend_y - 4:.3f}" x2="{legend_x + 18:.3f}" y2="{legend_y - 4:.3f}" '
               f'stroke="#7a3fe0" stroke-dasharray="5 3" stroke-width="1.2"/>')
    svg.append(label(legend_x + 24, legend_y + 16, "hinted", anchor="start"))
    svg.append(f'<line x1="{legend_x:.3f}" y1="{legend_y + 12:.3f}" x2="{legend_x + 18:.3f}" y2="{legend_y + 12:.3f}" '
               f'stroke="#000000" stroke-width="1.2"/>')

    svg.append("</svg>")
    Path(out_svg).write_text("\n".join(svg), encoding="utf-8")

    return {
        "char": char,
        "ppem": ppem,
        "dpi": dpi,
        "out_svg": str(Path(out_svg).resolve()),
        "hinted": {
            "lsb": px(hinted_metrics.horiBearingX),
            "advance": px(hinted_metrics.horiAdvance),
            "bbox_w": px(hinted_metrics.width),
            "bbox_h": px(hinted_metrics.height),
        },
        "original": {
            "lsb": px(orig_metrics.horiBearingX),
            "advance": px(orig_metrics.horiAdvance),
            "bbox_w": px(orig_metrics.width),
            "bbox_h": px(orig_metrics.height),
        },
        "ascender": asc,
        "descender": desc,
    }

if __name__ == "__main__":
    info = glyph_to_svg_compare("ARIALUNI.TTF", "a", ppem=12, dpi=96, hinting_target="mono", out_svg="a_hint.svg")
    print("Wrote:", info["out_svg"])
    print({k: v for k, v in info.items() if k != "out_svg"})
