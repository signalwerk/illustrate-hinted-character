# save as glyph_em_overlay.py
# pip install freetype-py pillow
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
# Pick a render mode for the bitmap layer
_RENDER_MODE = {
    "normal": ft.FT_RENDER_MODE_NORMAL,   # 8-bit gray
    "light":  ft.FT_RENDER_MODE_NORMAL,
    "mono":   ft.FT_RENDER_MODE_MONO,     # 1-bit
    "lcd":    ft.FT_RENDER_MODE_NORMAL,   # (true LCD needs filtering)
    "lcd_v":  ft.FT_RENDER_MODE_NORMAL,
}

def _png_data_uri_from_bitmap(bmp: ft.Bitmap) -> str:
    """Convert an FT_Bitmap (GRAY or MONO) to a PNG data URI with alpha only."""
    w, h, pitch, mode = bmp.width, bmp.rows, bmp.pitch, bmp.pixel_mode
    if w == 0 or h == 0:
        # empty
        return "data:image/png;base64," + b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")

    if mode == ft.FT_PIXEL_MODE_GRAY:
        buf = bmp.buffer
        alpha = bytearray(w * h)
        for y in range(h):
            row = buf[y * pitch : y * pitch + w]
            alpha[y * w : (y + 1) * w] = row
    elif mode == ft.FT_PIXEL_MODE_MONO:
        buf = bmp.buffer
        alpha = bytearray(w * h)
        for y in range(h):
            row_off = y * pitch
            for x in range(w):
                b = buf[row_off + (x >> 3)]
                alpha[y * w + x] = 255 if (b >> (7 - (x & 7))) & 1 else 0
    else:
        # treat unsupported modes as empty
        alpha = bytearray([0] * (w * h))

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    img.putalpha(Image.frombytes("L", (w, h), bytes(alpha)))
    bio = BytesIO()
    img.save(bio, format="PNG")
    return "data:image/png;base64," + b64encode(bio.getvalue()).decode("ascii")

def glyph_em_overlay(
    font_path: str,
    char: str,
    ppem: int = 20,
    hinting_target: str = "normal",
    use_hinted_bitmap: bool = True,  # whether to render bitmap from hinted or unhinted outline
    # out_svg: str = "glyph_em.svg",
    # visual tuning (all in EM units, proportional to UPEM by default)
    label_scale: float = 0.040,   # label font-size = UPEM * label_scale
    stroke_main: float = 0.006,   # main outline stroke = UPEM * stroke_main
    stroke_guides: float = 0.004, # guides stroke = UPEM * stroke_guides
    margin_em: float = 0.01,      # margins as a fraction of UPEM
    bitmap_style: str = "image",  # "image", "rects", "circles"
    bitmap_opacity: float = 1,  # opacity for bitmap layer (0.0 – 1.0)
):
    """
    SVG in EM units:
      - Original (unscaled, unhinted) glyph + metrics drawn in EM space.
      - Hinted outline converted back to EM units and overlaid.
      - FT-rendered bitmap (hinted or unhinted based on use_hinted_bitmap) scaled/translated into EM units.

    Args:
        use_hinted_bitmap: If True, bitmap is rendered from hinted outline. 
                          If False, bitmap is rendered from unhinted outline.

    All labels/line widths scale with UPEM (stable regardless of ppem).
    """
    face = ft.Face(font_path)
    # UPEM (often 2048; note the common value is 2048, not 2024)
    upem = face.units_per_EM

    # --- ORIGINAL (EM-space) -----------------------------------------------
    # Load NO_SCALE|NO_HINTING: outline and metrics in *font units* (EM units).
    face.load_char(char, ft.FT_LOAD_NO_HINTING | ft.FT_LOAD_NO_SCALE)
    orig_slot = face.glyph
    orig_outline = orig_slot.outline
    om = orig_slot.metrics  # EM units here

    # EM metrics (no /64 needed under NO_SCALE)
    lsb_em   = float(om.horiBearingX)
    top_em   = float(om.horiBearingY)
    width_em = float(om.width)
    height_em= float(om.height)
    adv_em   = float(om.horiAdvance)
    # Face vertical metrics in EM units
    asc_em   = float(face.ascender)
    desc_em  = float(face.descender)  # negative

    # Helpers for EM-space SVG path (y-down flip at write time)
    def em_path_from_outline(outline):
        cmds, open_flag = [], False
        def move_to(p, _):
            nonlocal open_flag
            if open_flag: cmds.append("Z")
            cmds.append(f"M {p.x:.3f} {-p.y:.3f}")
            open_flag = True
        def line_to(p, _):
            cmds.append(f"L {p.x:.3f} {-p.y:.3f}")
        def conic_to(c, p, _):
            cmds.append(f"Q {c.x:.3f} {-c.y:.3f} {p.x:.3f} {-p.y:.3f}")
        def cubic_to(c1, c2, p, _):
            cmds.append(
                "C "
                f"{c1.x:.3f} {-c1.y:.3f} "
                f"{c2.x:.3f} {-c2.y:.3f} "
                f"{p.x:.3f} {-p.y:.3f}"
            )
        outline.decompose(move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
        if open_flag: cmds.append("Z")
        return " ".join(cmds)

    orig_path_em = em_path_from_outline(orig_outline)

    # --- HINTED (pixel space) -> back to EM --------------------------------
    # Set pixel size for hinting and bitmap
    face.set_pixel_sizes(ppem, 0)
    # Load hinted scalable outline (26.6 pixel units in glyph.outline)
    flags_hinted = ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
    face.load_char(char, flags_hinted)
    hinted_slot = face.glyph
    hm = hinted_slot.metrics  # 26.6 pixels
    x_ppem = face.size.x_ppem
    y_ppem = face.size.y_ppem

    # Pixel->EM conversion factors
    # px_float * (UPEM / ppem) => EM
    def px_to_em_x(px26_6): return (px26_6 / 64.0) * (upem / float(x_ppem))
    def px_to_em_y(px26_6): return (px26_6 / 64.0) * (upem / float(y_ppem))

    def hinted_em_path(outline):
        cmds, open_flag = [], False
        def move_to(p, _):
            nonlocal open_flag
            if open_flag: cmds.append("Z")
            cmds.append(f"M {px_to_em_x(p.x):.3f} {-px_to_em_y(p.y):.3f}")
            open_flag = True
        def line_to(p, _):
            cmds.append(f"L {px_to_em_x(p.x):.3f} {-px_to_em_y(p.y):.3f}")
        def conic_to(c, p, _):
            cmds.append(
                f"Q {px_to_em_x(c.x):.3f} {-px_to_em_y(c.y):.3f} "
                f"{px_to_em_x(p.x):.3f} {-px_to_em_y(p.y):.3f}"
            )
        def cubic_to(c1, c2, p, _):
            cmds.append(
                "C "
                f"{px_to_em_x(c1.x):.3f} {-px_to_em_y(c1.y):.3f} "
                f"{px_to_em_x(c2.x):.3f} {-px_to_em_y(c2.y):.3f} "
                f"{px_to_em_x(p.x):.3f} {-px_to_em_y(p.y):.3f}"
            )
        outline.decompose(move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
        if open_flag: cmds.append("Z")
        return " ".join(cmds)

    hinted_path_em = hinted_em_path(hinted_slot.outline)

    # --- FT bitmap (hinted or unhinted), scaled & aligned in EM -------------
    render_mode = _RENDER_MODE[hinting_target]
    
    if use_hinted_bitmap:
        # Use hinted outline for bitmap
        hinted_slot.render(render_mode)
        bmp = hinted_slot.bitmap
        bmp_left_px = hinted_slot.bitmap_left
        bmp_top_px  = hinted_slot.bitmap_top
        # Hinted bearings in *pixel* units (floats)
        lsb_px  = hm.horiBearingX / 64.0
        top_px  = hm.horiBearingY / 64.0
    else:
        # Use unhinted outline for bitmap at the same pixel size
        # Don't call set_pixel_sizes again - it was already set above
        flags_unhinted = ft.FT_LOAD_NO_HINTING | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
        face.load_char(char, flags_unhinted)
        unhinted_slot = face.glyph
        unhinted_slot.render(render_mode)
        bmp = unhinted_slot.bitmap
        bmp_left_px = unhinted_slot.bitmap_left
        bmp_top_px  = unhinted_slot.bitmap_top
        # Unhinted bearings in *pixel* units (floats)
        um = unhinted_slot.metrics  # 26.6 pixels
        lsb_px  = um.horiBearingX / 64.0
        top_px  = um.horiBearingY / 64.0

    # Base image position in EM (from integer bitmap offsets)
    img_x_em = bmp_left_px * (upem / float(x_ppem))
    img_y_em = -bmp_top_px  * (upem / float(y_ppem))  # SVG y-down

    # Fractional compensation so bitmap aligns to the target outline exactly
    if use_hinted_bitmap:
        # Align hinted bitmap to hinted bearings (to align with hinted outline)
        dx_px = lsb_px - bmp_left_px
        dy_px = bmp_top_px - top_px
        dx_em = dx_px * (upem / float(x_ppem))
        dy_em = dy_px * (upem / float(y_ppem))
    else:
        # For unhinted bitmap: no compensation needed at all
        # Let FreeType position the bitmap naturally according to the unhinted outline
        dx_px = 0.0  # No horizontal compensation
        dy_px = 0.0  # No vertical compensation 
        dx_em = 0.0
        dy_em = 0.0

    # Image size in EM
    img_w_em = bmp.width * (upem / float(x_ppem))
    img_h_em = bmp.rows  * (upem / float(y_ppem))

    png_uri = _png_data_uri_from_bitmap(bmp)

    # --- ViewBox in EM units (margin relative to UPEM) ----------------------
    margin = margin_em * upem
    # Include original metrics (EM) and hinted bitmap extents
    img_left_em  = img_x_em + dx_em
    img_right_em = img_left_em + img_w_em
    left  = min(0.0, lsb_em, img_left_em) - margin
    right = max(adv_em, lsb_em + width_em, img_right_em) + margin
    top_vb    = -(asc_em + margin)     # SVG y-down
    height_vb = (asc_em - desc_em) + 2 * margin
    width_vb  = right - left

    # --- Scaled styles in EM units (proportional to UPEM) -------------------
    fs = label_scale * upem
    sw_main = stroke_main * upem
    sw_guid = stroke_guides * upem
    dash_a = 0.020 * upem
    dash_b = 0.015 * upem

    def label(x, y, text, anchor="start"):
        return (f'<text x="{x:.3f}" y="{y:.3f}" font-size="{fs:.3f}" '
                f'text-anchor="{anchor}" font-family="monospace">{text}</text>')

    # --- Build SVG ----------------------------------------------------------
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{left:.3f} {top_vb:.3f} {width_vb:.3f} {height_vb:.3f}" '
        f'width="{width_vb:.0f}" height="{height_vb:.0f}">'
    )
    svg.append('<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="#f9f9fb"/>'
               .format(left, top_vb, width_vb, height_vb))

    # ===== Lowest layer: hinted FT bitmap (in EM) ===========================
    if bitmap_style == "image":
        # Only include opacity in style if it's not 1.0
        opacity_style = f" opacity: {bitmap_opacity:.3f};" if bitmap_opacity < 1.0 else ""
        svg.append(
            f'<image x="{img_x_em:.3f}" y="{img_y_em:.3f}" '
            f'width="{img_w_em:.3f}" height="{img_h_em:.3f}" '
            f'href="{png_uri}" transform="translate({dx_em:.3f},{dy_em:.3f})" '
            f'style="image-rendering: pixelated;{opacity_style}"/>'
        )
    else:
        # Recreate alpha buffer as grayscale values
        alpha = bmp.buffer
        w, h, pitch = bmp.width, bmp.rows, bmp.pitch
        px_w = upem / float(x_ppem)
        px_h = upem / float(y_ppem)
        for y in range(h):
            row = alpha[y * pitch : y * pitch + w] if bmp.pixel_mode == ft.FT_PIXEL_MODE_GRAY else None
            for x in range(w):
                if bmp.pixel_mode == ft.FT_PIXEL_MODE_GRAY:
                    a = row[x]
                elif bmp.pixel_mode == ft.FT_PIXEL_MODE_MONO:
                    b = alpha[y * pitch + (x >> 3)]
                    a = 255 if (b >> (7 - (x & 7))) & 1 else 0
                else:
                    a = 0
                if a == 0:
                    continue  # Skip transparent
                cx = img_x_em + (x + dx_px) * px_w
                cy = img_y_em + (y + dy_px) * px_h
                size_x = px_w
                size_y = px_h
                # Pixel darkness controls color intensity (0=white, 255=black)
                intensity = a / 255.0
                gray_value = int(255 * (1.0 - intensity))  # Invert: high alpha = dark color
                fill_color = f"rgb({gray_value},{gray_value},{gray_value})"
                
                # Only include fill-opacity if it's not 1.0
                opacity_attr = f' fill-opacity="{bitmap_opacity:.3f}"' if bitmap_opacity < 1.0 else ""
                
                if bitmap_style == "rects":
                    svg.append(
                        f'<rect x="{cx:.3f}" y="{cy:.3f}" width="{size_x:.3f}" height="{size_y:.3f}" '
                        f'fill="{fill_color}"{opacity_attr}/>'
                    )
                elif bitmap_style == "circles":
                    r = px_w * 0.5
                    svg.append(
                        f'<circle cx="{cx + r:.3f}" cy="{cy + r:.3f}" r="{r:.3f}" '
                        f'fill="{fill_color}"{opacity_attr}/>'
                    )

    # Optional soft fill of hinted vector (in EM) just above bitmap
    svg.append(f'<path d="{hinted_path_em}" fill="#000" fill-opacity="0.05" stroke="none"/>')

    # Guides (EM space)
    svg.append(
        f'<line x1="{left:.3f}" y1="{0:.3f}" x2="{right:.3f}" y2="{0:.3f}" '
        f'stroke="#c0c0c0" stroke-width="{sw_guid:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(
        f'<line x1="{left:.3f}" y1="{-asc_em:.3f}" x2="{right:.3f}" y2="{-asc_em:.3f}" '
        f'stroke="#e0a000" stroke-width="{sw_guid:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(
        f'<line x1="{left:.3f}" y1="{-desc_em:.3f}" x2="{right:.3f}" y2="{-desc_em:.3f}" '
        f'stroke="#e0a000" stroke-width="{sw_guid:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(left + 0.02 * upem, -asc_em - 0.02 * upem, "ascender"))
    svg.append(label(left + 0.02 * upem, -0.02 * upem, "baseline"))
    svg.append(label(left + 0.02 * upem, -desc_em + 0.06 * upem, "descender"))

    # Bearings & advance (from ORIGINAL EM metrics)
    svg.append(
        f'<line x1="0" y1="{-asc_em:.3f}" x2="0" y2="{-desc_em:.3f}" '
        f'stroke="#c0c0c0" stroke-width="{sw_guid:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(0, -asc_em - 0.04 * upem, "origin x=0", anchor="middle"))
    svg.append(
        f'<line x1="{lsb_em:.3f}" y1="{-asc_em:.3f}" x2="{lsb_em:.3f}" y2="{-desc_em:.3f}" '
        f'stroke="#00a0e0" stroke-width="{sw_guid:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(lsb_em, -asc_em - 0.04 * upem, f"LSB={lsb_em:.1f} em-units", anchor="middle"))
    svg.append(
        f'<line x1="{adv_em:.3f}" y1="{-asc_em:.3f}" x2="{adv_em:.3f}" y2="{-desc_em:.3f}" '
        f'stroke="#008000" stroke-width="{sw_guid:.3f}" '
        f'stroke-dasharray="{dash_a:.3f} {dash_b:.3f}"/>'
    )
    svg.append(label(adv_em, -asc_em - 0.04 * upem, f"advance={adv_em:.1f} em-units", anchor="middle"))

    # Original glyph bbox (EM metrics)
    svg.append(
        f'<rect x="{lsb_em:.3f}" y="{-top_em:.3f}" width="{width_em:.3f}" height="{height_em:.3f}" '
        f'fill="none" stroke="#cc3333" stroke-width="{sw_guid:.3f}"/>'
    )

    svg.append(label(lsb_em + 0.02 * upem, -top_em + 0.06 * upem,
                     f"bbox {width_em:.0f}×{height_em:.0f} em-units"))

    # --- Outlines -----------------------------------------------------------
    # Original (EM) — dashed purple
    svg.append(
        f'<path d="{orig_path_em}" fill="none" stroke="#7a3fe0" '
        f'stroke-width="{sw_main:.3f}" stroke-dasharray="{dash_a:.3f} {dash_b:.3f}" opacity="0.95"/>'
    )
    # Hinted outline mapped back to EM — solid black
    svg.append(
        f'<path d="{hinted_path_em}" fill="none" stroke="#000000" '
        f'stroke-width="{sw_main:.3f}"/>'
    )

    # Legend
    lx = left + 0.04 * upem
    ly = -asc_em + 0.10 * upem
    svg.append(label(lx + 0.12 * upem, ly, "original (EM)", anchor="start"))
    svg.append(
        f'<line x1="{lx:.3f}" y1="{ly - 0.02 * upem:.3f}" '
        f'x2="{lx + 0.10 * upem:.3f}" y2="{ly - 0.02 * upem:.3f}" '
        f'stroke="#7a3fe0" stroke-dasharray="{dash_a:.3f} {dash_b:.3f}" '
        f'stroke-width="{sw_main:.3f}"/>'
    )
    svg.append(label(lx + 0.12 * upem, ly + 0.08 * upem, "hinted outline (EM)", anchor="start"))
    svg.append(
        f'<line x1="{lx:.3f}" y1="{ly + 0.06 * upem:.3f}" '
        f'x2="{lx + 0.10 * upem:.3f}" y2="{ly + 0.06 * upem:.3f}" '
        f'stroke="#000000" stroke-width="{sw_main:.3f}"/>'
    )
    bitmap_type = "hinted" if use_hinted_bitmap else "unhinted"
    svg.append(label(lx + 0.12 * upem, ly + 0.16 * upem, f"FT bitmap ({bitmap_type})", anchor="start"))
    svg.append(
        f'<rect x="{lx:.3f}" y="{ly + 0.12 * upem:.3f}" '
        f'width="{0.08 * upem:.3f}" height="{0.05 * upem:.3f}" '
        f'fill="#000" fill-opacity="0.4" stroke="none"/>'
    )

    svg.append("</svg>")
    bitmap_suffix = "hint" if use_hinted_bitmap else "unhint"
    out_svg = f"{char}_{hinting_target}_{bitmap_suffix}.svg"
    Path(out_svg).write_text("\n".join(svg), encoding="utf-8")

    return {
        "out_svg": str(Path(out_svg).resolve()),
        "upem": upem,
        "ppem": ppem,
        "x_ppem": x_ppem,
        "y_ppem": y_ppem,
        "original_metrics_em": {
            "lsb": lsb_em, "top": top_em, "width": width_em, "height": height_em, "advance": adv_em,
            "ascender": asc_em, "descender": desc_em,
        },
        "bitmap_align": {
            "bitmap_left_px": bmp_left_px, "bitmap_top_px": bmp_top_px,
            "dx_px": dx_px, "dy_px": dy_px, "dx_em": dx_em, "dy_em": dy_em,
        },
    }

if __name__ == "__main__":
    # Generate with hinted bitmap
    info_hinted_mono = glyph_em_overlay(
        "ARIALUNI.TTF", "a",
        ppem=12, hinting_target="mono",
        use_hinted_bitmap=True,
        bitmap_style="circles",
        bitmap_opacity=0.25
    )
    print("Wrote (hinted bitmap):", info_hinted_mono["out_svg"])
    
    # Generate with unhinted bitmap
    info_unhinted_mono = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="mono",
        use_hinted_bitmap=False,
        bitmap_style="circles",
        bitmap_opacity=0.25
    )
    print("Wrote (unhinted bitmap):", info_unhinted_mono["out_svg"])

    info_hinted_normal = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="normal",
        use_hinted_bitmap=True,
        bitmap_style="rects",
    )
    print("Wrote (hinted bitmap):", info_hinted_normal["out_svg"])

    info_unhinted_normal = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="normal",
        use_hinted_bitmap=False,
        bitmap_style="rects",
    )
    print("Wrote (unhinted bitmap):", info_unhinted_normal["out_svg"])
    
    # print({k: v for k in info_hinted_normal.items() if k != "out_svg"})
