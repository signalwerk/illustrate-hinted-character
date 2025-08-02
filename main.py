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
    layers: str = "path-original,path-hinted,bitmap-hinted,guides,bearings,bbox,labels",  # comma-separated list of layers
    # out_svg: str = "glyph_em.svg",
    # visual tuning (all in EM units, proportional to UPEM by default)
    label_scale: float = 0.040,   # label font-size = UPEM * label_scale
    stroke_main: float = 0.003,   # main outline stroke = UPEM * stroke_main
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
        layers: Comma-separated list of layers to include. Available layers:
                - path-original: Original unhinted outline (dashed purple)
                - path-hinted: Hinted outline (solid black)
                - bitmap-original: Bitmap from unhinted outline
                - bitmap-hinted: Bitmap from hinted outline
                - guides: ascender, descender lines
                - baseline: Baseline line
                - bearings: LSB and advance lines
                - bbox: Bounding box rectangle
                - labels: Text labels and legend

    All labels/line widths scale with UPEM (stable regardless of ppem).
    """
    face = ft.Face(font_path)
    # UPEM (often 2048; note the common value is 2048, not 2024)
    upem = face.units_per_EM

    # Parse layers
    layer_set = set(layer.strip() for layer in layers.split(",") if layer.strip())
    
    # Determine what bitmaps we need to generate
    need_hinted_bitmap = "bitmap-hinted" in layer_set
    need_unhinted_bitmap = "bitmap-original" in layer_set
    need_any_bitmap = need_hinted_bitmap or need_unhinted_bitmap

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

    # --- FT bitmap (hinted and/or unhinted), scaled & aligned in EM --------
    render_mode = _RENDER_MODE[hinting_target]
    bitmaps = {}
    
    if need_hinted_bitmap:
        # Generate hinted bitmap
        hinted_slot.render(render_mode)
        hinted_bmp = hinted_slot.bitmap
        hinted_bmp_left_px = hinted_slot.bitmap_left
        hinted_bmp_top_px  = hinted_slot.bitmap_top
        # Hinted bearings in *pixel* units (floats)
        hinted_lsb_px  = hm.horiBearingX / 64.0
        hinted_top_px  = hm.horiBearingY / 64.0
        
        # Align hinted bitmap to hinted bearings
        hinted_dx_px = hinted_lsb_px - hinted_bmp_left_px
        hinted_dy_px = hinted_bmp_top_px - hinted_top_px
        hinted_dx_em = hinted_dx_px * (upem / float(x_ppem))
        hinted_dy_em = hinted_dy_px * (upem / float(y_ppem))
        
        bitmaps['hinted'] = {
            'bitmap': hinted_bmp,
            'img_x_em': hinted_bmp_left_px * (upem / float(x_ppem)),
            'img_y_em': -hinted_bmp_top_px * (upem / float(y_ppem)),
            'img_w_em': hinted_bmp.width * (upem / float(x_ppem)),
            'img_h_em': hinted_bmp.rows * (upem / float(y_ppem)),
            'dx_em': hinted_dx_em,
            'dy_em': hinted_dy_em,
            'png_uri': _png_data_uri_from_bitmap(hinted_bmp)
        }
    
    if need_unhinted_bitmap:
        # Generate unhinted bitmap at the same pixel size
        flags_unhinted = ft.FT_LOAD_NO_HINTING | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
        face.load_char(char, flags_unhinted)
        unhinted_slot = face.glyph
        unhinted_slot.render(render_mode)
        unhinted_bmp = unhinted_slot.bitmap
        unhinted_bmp_left_px = unhinted_slot.bitmap_left
        unhinted_bmp_top_px  = unhinted_slot.bitmap_top
        
        # For unhinted bitmap: no compensation needed
        unhinted_dx_em = 0.0
        unhinted_dy_em = 0.0
        
        bitmaps['unhinted'] = {
            'bitmap': unhinted_bmp,
            'img_x_em': unhinted_bmp_left_px * (upem / float(x_ppem)),
            'img_y_em': -unhinted_bmp_top_px * (upem / float(y_ppem)),
            'img_w_em': unhinted_bmp.width * (upem / float(x_ppem)),
            'img_h_em': unhinted_bmp.rows * (upem / float(y_ppem)),
            'dx_em': unhinted_dx_em,
            'dy_em': unhinted_dy_em,
            'png_uri': _png_data_uri_from_bitmap(unhinted_bmp)
        }

    # For viewbox calculation, use any available bitmap or fallback to hinted metrics
    if bitmaps:
        # Use the first available bitmap for viewbox calculation
        primary_bitmap = list(bitmaps.values())[0]
        img_left_em = primary_bitmap['img_x_em'] + primary_bitmap['dx_em']
        img_right_em = img_left_em + primary_bitmap['img_w_em']
    else:
        # No bitmaps requested, use zero extents
        img_left_em = img_right_em = 0.0

    # --- ViewBox in EM units (margin relative to UPEM) ----------------------
    margin = margin_em * upem
    # Include original metrics (EM) and bitmap extents if any
    left  = min(0.0, lsb_em, img_left_em) - margin
    right = max(adv_em, lsb_em + width_em, img_right_em) + margin
    top_vb    = -(asc_em + margin)     # SVG y-down
    height_vb = (asc_em - desc_em) + 2 * margin
    width_vb  = right - left

    # --- Scaled styles in EM units (proportional to UPEM) -------------------
    fs = label_scale * upem
    sw_main = stroke_main * upem
    sw_guid = stroke_guides * upem
    dash_a = 0.0 * upem
    dash_b = 0.007 * upem


    # --- styles for paths 
    path_original_style = f'stroke="#000" stroke-width="{sw_main:.3f}"'
    path_hinted_style = f'stroke="#0054a2" stroke-width="{sw_main*1.5:.3f}" stroke-dasharray="{dash_a:.3f} {dash_b:.3f}" stroke-linecap="round"'

    

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
    svg.append('<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="#fff"/>'
               .format(left, top_vb, width_vb, height_vb))

    # Helper function to render bitmap
    def render_bitmap(bitmap_info, bitmap_type):
        bmp = bitmap_info['bitmap']
        img_x_em = bitmap_info['img_x_em']
        img_y_em = bitmap_info['img_y_em']
        img_w_em = bitmap_info['img_w_em']
        img_h_em = bitmap_info['img_h_em']
        dx_em = bitmap_info['dx_em']
        dy_em = bitmap_info['dy_em']
        png_uri = bitmap_info['png_uri']
        
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
            dx_px = dx_em / (upem / float(x_ppem))
            dy_px = dy_em / (upem / float(y_ppem))
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

    # ===== Bitmap layers (lowest layer) ====================================
    # Render bitmaps in order: unhinted first (if present), then hinted
    if "bitmap-original" in layer_set and "unhinted" in bitmaps:
        render_bitmap(bitmaps["unhinted"], "unhinted")
    
    if "bitmap-hinted" in layer_set and "hinted" in bitmaps:
        render_bitmap(bitmaps["hinted"], "hinted")

    # ===== Guides (EM space) ===============================================
    if "guides" in layer_set:

        svg.append(
            f'<line x1="{left:.3f}" y1="{-asc_em:.3f}" x2="{right:.3f}" y2="{-asc_em:.3f}" '
            f'stroke="#e0a000" stroke-width="{sw_guid:.3f}"/>'
        )
        svg.append(
            f'<line x1="{left:.3f}" y1="{-desc_em:.3f}" x2="{right:.3f}" y2="{-desc_em:.3f}" '
            f'stroke="#e0a000" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(left + 0.02 * upem, -asc_em - 0.02 * upem, "ascender"))
            svg.append(label(left + 0.02 * upem, -desc_em + 0.06 * upem, "descender"))

    if "baseline" in layer_set:
        svg.append(
            f'<line x1="{left:.3f}" y1="{0:.3f}" x2="{right:.3f}" y2="{0:.3f}" '
            f'stroke="#c0c0c0" stroke-width="{sw_guid:.3f}"/>'
        )
        svg.append(label(left + 0.02 * upem, -0.02 * upem, "baseline"))

    # ===== Bearings & advance (from ORIGINAL EM metrics) ===================
    if "bearings" in layer_set:
        svg.append(
            f'<line x1="0" y1="{-asc_em:.3f}" x2="0" y2="{-desc_em:.3f}" '
            f'stroke="#c0c0c0" stroke-width="{sw_guid:.3f}"/>'
        )
        svg.append(
            f'<line x1="{lsb_em:.3f}" y1="{-asc_em:.3f}" x2="{lsb_em:.3f}" y2="{-desc_em:.3f}" '
            f'stroke="#00a0e0" stroke-width="{sw_guid:.3f}"/>'
        )
        svg.append(
            f'<line x1="{adv_em:.3f}" y1="{-asc_em:.3f}" x2="{adv_em:.3f}" y2="{-desc_em:.3f}" '
            f'stroke="#008000" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(0, -asc_em - 0.04 * upem, "origin x=0", anchor="middle"))
            svg.append(label(lsb_em, -asc_em - 0.04 * upem, f"LSB={lsb_em:.1f} em-units", anchor="middle"))
            svg.append(label(adv_em, -asc_em - 0.04 * upem, f"advance={adv_em:.1f} em-units", anchor="middle"))

    # ===== Original glyph bbox (EM metrics) ================================
    if "bbox" in layer_set:
        svg.append(
            f'<rect x="{lsb_em:.3f}" y="{-top_em:.3f}" width="{width_em:.3f}" height="{height_em:.3f}" '
            f'fill="none" stroke="#cc3333" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(lsb_em + 0.02 * upem, -top_em + 0.06 * upem,
                             f"bbox {width_em:.0f}×{height_em:.0f} em-units"))

    # ===== Outlines ========================================================
    # Original (EM) — dashed purple
    if "path-original" in layer_set:
        svg.append(
            f'<path d="{orig_path_em}" fill="none" {path_original_style} />'
        )
    # Hinted outline mapped back to EM — solid black
    if "path-hinted" in layer_set:
        svg.append(
            f'<path d="{hinted_path_em}" fill="none" {path_hinted_style} />'
        )

    # ===== Legend ==========================================================
    if "labels" in layer_set:
        lx = left + 0.04 * upem
        ly = -asc_em + 0.10 * upem
        
        if "path-original" in layer_set:
            svg.append(label(lx + 0.12 * upem, ly, "original (EM)", anchor="start"))
            svg.append(
                f'<line x1="{lx:.3f}" y1="{ly - 0.02 * upem:.3f}" '
                f'x2="{lx + 0.10 * upem:.3f}" y2="{ly - 0.02 * upem:.3f}" '
                f'{path_original_style} />'
            )
            ly += 0.08 * upem
            
        if "path-hinted" in layer_set:
            svg.append(label(lx + 0.12 * upem, ly, "hinted outline (EM)", anchor="start"))
            svg.append(
                f'<line x1="{lx:.3f}" y1="{ly - 0.02 * upem:.3f}" '
                f'x2="{lx + 0.10 * upem:.3f}" y2="{ly - 0.02 * upem:.3f}" '
                f'{path_hinted_style} />'
            )
            ly += 0.08 * upem
            
        if "bitmap-hinted" in layer_set:
            svg.append(label(lx + 0.12 * upem, ly, "FT bitmap (hinted)", anchor="start"))
            svg.append(
                f'<rect x="{lx:.3f}" y="{ly - 0.02 * upem:.3f}" '
                f'width="{0.08 * upem:.3f}" height="{0.05 * upem:.3f}" '
                f'fill="#000" fill-opacity="0.4" stroke="none"/>'
            )
            ly += 0.08 * upem
            
        if "bitmap-original" in layer_set:
            svg.append(label(lx + 0.12 * upem, ly, "FT bitmap (unhinted)", anchor="start"))
            svg.append(
                f'<rect x="{lx:.3f}" y="{ly - 0.02 * upem:.3f}" '
                f'width="{0.08 * upem:.3f}" height="{0.05 * upem:.3f}" '
                f'fill="#000" fill-opacity="0.6" stroke="none"/>'
            )

    svg.append("</svg>")
    
    # Generate output filename based on layers
    layer_suffix = "_".join(sorted(layer_set)) if layer_set else "empty"
    layer_suffix = layer_suffix.replace("-", "").replace(",", "_")  # Clean up filename
    out_svg = f"{char}_{hinting_target}_{layer_suffix}.svg"
    Path(out_svg).write_text("\n".join(svg), encoding="utf-8")

    return {
        "out_svg": str(Path(out_svg).resolve()),
        "upem": upem,
        "ppem": ppem,
        "x_ppem": x_ppem,
        "y_ppem": y_ppem,
        "layers": list(layer_set),
        "original_metrics_em": {
            "lsb": lsb_em, "top": top_em, "width": width_em, "height": height_em, "advance": adv_em,
            "ascender": asc_em, "descender": desc_em,
        },
        "bitmaps_generated": list(bitmaps.keys()),
    }

if __name__ == "__main__":
    info_1 = glyph_em_overlay(
        "ARIALUNI.TTF", "a",
        ppem=12, hinting_target="mono",
        layers="path-original,bitmap-original,labels",
        bitmap_style="circles",
        bitmap_opacity=0.25
    )
    print("Wrote (example 1):", info_1["out_svg"])
    
    info_2 = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="mono",
        layers="path-original,path-hinted,bitmap-hinted,labels",
        bitmap_style="circles",
        bitmap_opacity=0.25
    )
    print("Wrote (example 2):", info_2["out_svg"])

    info_full = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="normal",
        layers="path-original,bitmap-original,labels",
        bitmap_style="rects",
    )
    print("Wrote (full comparison):", info_full["out_svg"])

    info_outlines = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="normal",
        layers="path-original,path-hinted,bitmap-hinted,labels",
    )
    print("Wrote (outlines only):", info_outlines["out_svg"])

    info_debug = glyph_em_overlay(
        "ARIALUNI.TTF", "a", 
        ppem=12, hinting_target="normal",
        layers="path-original,path-hinted,bitmap-hinted,guides,baseline,bearings,bbox,labels",
    )
    print("Wrote (debug):", info_debug["out_svg"])
    
