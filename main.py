# save as glyph_to_svg_compare.py
import freetype as ft
from pathlib import Path
from io import BytesIO
from base64 import b64encode
from PIL import Image

_LOAD_TARGET = {
    "normal": ft.FT_LOAD_TARGET_NORMAL,
    "light":  ft.FT_LOAD_TARGET_LIGHT,
    "mono":   ft.FT_LOAD_TARGET_MONO,
    "lcd":    ft.FT_LOAD_TARGET_LCD,
    "lcd_v":  ft.FT_LOAD_TARGET_LCD_V,
}

# Map hint target to a render mode for the bitmap layer
# (For LCD/LCD_V we default to grayscale; true subpixel needs device filtering.)
_RENDER_MODE = {
    "normal": ft.FT_RENDER_MODE_NORMAL,
    "light":  ft.FT_RENDER_MODE_NORMAL,
    "mono":   ft.FT_RENDER_MODE_MONO,
    "lcd":    ft.FT_RENDER_MODE_NORMAL,
    "lcd_v":  ft.FT_RENDER_MODE_NORMAL,
}

def _bitmap_to_png_data_uri(bmp: ft.Bitmap) -> str:
    """Convert a FreeType FT_Bitmap to a PNG data URI (black glyph on transparent)."""
    w, h = bmp.width, bmp.rows
    pitch = bmp.pitch
    mode = bmp.pixel_mode

    # Build an 8-bit alpha mask from FT bitmap buffer
    if mode == ft.FT_PIXEL_MODE_GRAY:
        # Buffer is rows * pitch; each pixel 0..255 coverage.
        # Copy row-by-row to strip pitch padding.
        alpha = bytearray(w * h)
        buf = bmp.buffer
        for y in range(h):
            row = buf[y * pitch : y * pitch + w]
            alpha[y * w : (y + 1) * w] = row
    elif mode == ft.FT_PIXEL_MODE_MONO:
        # Packed 1bpp, MSB first per byte.
        buf = bmp.buffer
        alpha = bytearray(w * h)
        for y in range(h):
            row_off = y * pitch
            for x in range(w):
                b = buf[row_off + (x >> 3)]
                bit = 7 - (x & 7)
                alpha[y * w + x] = 255 if ((b >> bit) & 1) else 0
    else:
        # Fallback: treat as empty
        alpha = bytearray([0] * (w * h))

    # Compose RGBA: black glyph with alpha mask
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.putalpha(Image.frombytes("L", (w, h), bytes(alpha)))

    bio = BytesIO()
    img.save(bio, format="PNG")
    return "data:image/png;base64," + b64encode(bio.getvalue()).decode("ascii")

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
      • Hinted FT bitmap (lowest layer, from FT_Render_Glyph)
      • Filled hinted vector (soft background fill)
      • Original outline (same size, unhinted) — dashed purple
      • Hinted outline — solid black (on top)
    Plus metrics (baseline, ascender, descender, origin, LSB, advance, bbox).

    All strokes, dashes, and label font sizes scale in proportion to `ppem`.
    """

    face = ft.Face(font_path)
    face.set_char_size(0, ppem * 64, dpi, dpi)
    scale26_6 = 1.0 / 64.0

    def px(v):  # 26.6 -> float pixels
        return v * scale26_6

    # --- Style scaling (proportional to ppem) -------------------------------
    k = ppem / 64.0  # 1.0 at 64 ppem
    sw_main   = 1.2 * k         # outline strokes
    sw_guides = 1.0 * k         # guide strokes
    dash_a    = 4.0 * k         # dash pattern
    dash_b    = 3.0 * k
    font_sz   = 10.0 * k        # label font size
    margin    = 0.30 * ppem     # proportional margin

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

    # --- Render FT bitmap from the hinted glyph (lowest SVG layer) ----------
    # (Rendering uses the same hinted load; render mode chosen from hinting_target.)
    render_mode = _RENDER_MODE[hinting_target]
    hinted_slot.render(render_mode)
    bmp = hinted_slot.bitmap  # FT_Bitmap
    # Position: top-left of bitmap relative to baseline/origin
    bmp_x = hinted_slot.bitmap_left
    bmp_y = -hinted_slot.bitmap_top  # SVG y-down
    png_data_uri = _bitmap_to_png_data_uri(bmp)

    # --- Load original (unhinted) outline at the same size ------------------
    flags_unhinted = ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | ft.FT_LOAD_NO_HINTING
    face.load_char(char, flags_unhinted)
    orig_slot = face.glyph
    orig_path = outline_to_path(orig_slot.outline)
    orig_metrics = orig_slot.metrics  # for info

    # Metrics for guides (use hinted to match what actually renders)
    lsb    = px(hinted_metrics.horiBearingX)
    top    = px(hinted_metrics.horiBearingY)
    bbox_w = px(hinted_metrics.width)
    bbox_h = px(hinted_metrics.height)
    adv    = px(hinted_metrics.horiAdvance)

    # Correct asc/desc
    asc  = px(face.size.ascender)
    desc = px(face.size.descender)  # negative

    # ViewBox to include all guides with proportional margin
    left   = min(0.0, lsb, bmp_x) - margin
    right  = max(adv, lsb + bbox_w, bmp_x + bmp.width) + margin
    top_svg    = -(asc + margin)
    height_svg = (asc - desc) + 2 * margin
    width_svg  = right - left

    def label(x, y, text, anchor="start"):
        return (
            f'<text x="{x:.3f}" y="{y:.3f}" font-size="{font_sz:.3f}" '
            f'text-anchor="{anchor}" font-family="monospace">{text}</text>'
        )

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{left:.3f} {top_svg:.3f} {width_svg:.3f} {height_svg:.3f}" '
        f'width="{width_svg:.0f}" height="{height_svg:.0f}">'
    )

    # Background
    svg.append('<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="#f9f9fb"/>'
               .format(left, top_svg, width_svg, height_svg))

    # === LOWEST CONTENT LAYER: FT-rendered hinted bitmap ====================
    svg.append(
        f'<image x="{bmp_x:.3f}" y="{bmp_y:.3f}" width="{bmp.width:.3f}" height="{bmp.rows:.3f}" '
        f'href="{png_data_uri}" style="image-rendering: pixelated;"/>'
    )

    # Optional soft fill of hinted vector (still below guides/outlines)
    svg.append(f'<path d="{hinted_path}" fill="#000000" fill-opacity="0.05" stroke="none"/>')

    # Guides: baseline, ascender, descender
    svg.append(
        f'<line x1="{left:.3f}" y1="{0:.3f}" x2="{right:.3f}" y2="{0:.3f}" '
        f'stroke="#c0c0c0" stroke-width="{sw_guides:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(
        f'<line x1="{left:.3f}" y1="{-asc:.3f}" x2="{right:.3f}" y2="{-asc:.3f}" '
        f'stroke="#e0a000" stroke-width="{sw_guides:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(
        f'<line x1="{left:.3f}" y1="{-desc:.3f}" x2="{right:.3f}" y2="{-desc:.3f}" '
        f'stroke="#e0a000" stroke-width="{sw_guides:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(left + 3 * k, -asc - 3 * k, "ascender"))
    svg.append(label(left + 3 * k, -3 * k, "baseline"))
    svg.append(label(left + 3 * k, -desc + 12 * k, "descender"))

    # Origin, LSB, advance (from hinted metrics)
    svg.append(
        f'<line x1="0" y1="{-asc:.3f}" x2="0" y2="{-desc:.3f}" '
        f'stroke="#c0c0c0" stroke-width="{sw_guides:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(0, -asc - 6 * k, "origin x=0", anchor="middle"))
    svg.append(
        f'<line x1="{lsb:.3f}" y1="{-asc:.3f}" x2="{lsb:.3f}" y2="{-desc:.3f}" '
        f'stroke="#00a0e0" stroke-width="{sw_guides:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(lsb, -asc - 6 * k, f"LSB={lsb:.2f}px", anchor="middle"))
    svg.append(
        f'<line x1="{adv:.3f}" y1="{-asc:.3f}" x2="{adv:.3f}" y2="{-desc:.3f}" '
        f'stroke="#008000" stroke-width="{sw_guides:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(adv, -asc - 6 * k, f"advance={adv:.2f}px", anchor="middle"))

    # Glyph bbox (hinted metrics)
    svg.append(
        f'<rect x="{lsb:.3f}" y="{-top:.3f}" width="{bbox_w:.3f}" height="{bbox_h:.3f}" '
        f'fill="none" stroke="#cc3333" stroke-width="{sw_guides:.3f}"/>'
    )
    svg.append(label(lsb + 3 * k, -top + 12 * k, f"bbox {bbox_w:.2f}×{bbox_h:.2f}px"))

    # --- Outlines ---
    # Original (unhinted) outline — dashed purple
    svg.append(
        f'<path d="{orig_path}" fill="none" stroke="#7a3fe0" '
        f'stroke-width="{sw_main:.3f}" stroke-dasharray="{dash_a:.3f} {dash_b:.3f}" opacity="0.95"/>'
    )
    # Hinted outline — solid black on top
    svg.append(f'<path d="{hinted_path}" fill="none" stroke="#000000" stroke-width="{sw_main:.3f}"/>')

    # Legend
    legend_x = left + 8 * k
    legend_y = -asc + 16 * k
    svg.append(label(legend_x + 24 * k, legend_y, "original (unhinted)", anchor="start"))
    svg.append(
        f'<line x1="{legend_x:.3f}" y1="{legend_y - 4 * k:.3f}" '
        f'x2="{legend_x + 18 * k:.3f}" y2="{legend_y - 4 * k:.3f}" '
        f'stroke="#7a3fe0" stroke-dasharray="{dash_a:.3f} {dash_b:.3f}" '
        f'stroke-width="{sw_main:.3f}"/>'
    )
    svg.append(label(legend_x + 24 * k, legend_y + 16 * k, "hinted outline", anchor="start"))
    svg.append(
        f'<line x1="{legend_x:.3f}" y1="{legend_y + 12 * k:.3f}" '
        f'x2="{legend_x + 18 * k:.3f}" y2="{legend_y + 12 * k:.3f}" '
        f'stroke="#000000" stroke-width="{sw_main:.3f}"/>'
    )
    svg.append(label(legend_x + 24 * k, legend_y + 32 * k, "FT bitmap (hinted)", anchor="start"))
    svg.append(
        f'<rect x="{legend_x:.3f}" y="{legend_y + 24 * k:.3f}" width="{12 * k:.3f}" height="{8 * k:.3f}" '
        f'fill="#000" fill-opacity="0.4" stroke="none"/>'
    )

    svg.append("</svg>")
    Path(out_svg).write_text("\n".join(svg), encoding="utf-8")

    return {
        "char": char,
        "ppem": ppem,
        "dpi": dpi,
        "out_svg": str(Path(out_svg).resolve()),
        "hinted": {"lsb": lsb, "advance": adv, "bbox_w": bbox_w, "bbox_h": bbox_h},
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
    info = glyph_to_svg_compare(
        "ARIALUNI.TTF", "a", ppem=12, dpi=96,
        hinting_target="mono", out_svg="a_hint.svg"
    )
    print("Wrote:", info["out_svg"])
    print({k: v for k, v in info.items() if k != "out_svg"})
