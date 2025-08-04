"""
Font Hinting Visualization Tool

This script creates SVG visualizations that demonstrate how font hinting affects
glyph outlines and their bitmap rendering. It overlays original (unhinted) outlines,
hinted outlines, and bitmap representations to show the differences.

Dependencies: freetype-py, pillow
Usage: python main.py
"""

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

# print the freetype version
version = ".".join(map(str, ft.version()))
print(f"Freetype version: {version}")

def _png_data_uri_from_bitmap(bmp: ft.Bitmap) -> str:
    """
    Convert a FreeType bitmap to a PNG data URI for embedding in SVG.
    
    Handles both grayscale (FT_PIXEL_MODE_GRAY) and monochrome (FT_PIXEL_MODE_MONO) 
    bitmaps by converting them to RGBA images with transparency based on the 
    original pixel values.
    
    Args:
        bmp: FreeType bitmap object
        
    Returns:
        Base64-encoded PNG data URI suitable for SVG <image> elements
    """
    w, h, pitch, mode = bmp.width, bmp.rows, bmp.pitch, bmp.pixel_mode
    if w == 0 or h == 0:
        # empty
        return "data:image/png;base64," + b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")

    if mode == ft.FT_PIXEL_MODE_GRAY:
        # 8-bit grayscale: each byte represents one pixel's alpha value
        buf = bmp.buffer
        alpha = bytearray(w * h)
        for y in range(h):
            row = buf[y * pitch : y * pitch + w]
            alpha[y * w : (y + 1) * w] = row
    elif mode == ft.FT_PIXEL_MODE_MONO:
        # 1-bit monochrome: each bit represents one pixel (8 pixels per byte)
        buf = bmp.buffer
        alpha = bytearray(w * h)
        for y in range(h):
            row_off = y * pitch
            for x in range(w):
                byte_idx = row_off + (x >> 3)  # Which byte contains this pixel
                bit_idx = 7 - (x & 7)          # Which bit within that byte
                b = buf[byte_idx]
                alpha[y * w + x] = 255 if (b >> bit_idx) & 1 else 0
    else:
        # Unsupported pixel modes default to transparent
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
    output_filename: str = None,  # custom filename override (if None, auto-generates filename)
    # visual tuning (all in EM units, proportional to UPEM by default)
    label_scale: float = 0.040,   # label font-size = UPEM * label_scale
    stroke_main: float = 0.003,   # main outline stroke = UPEM * stroke_main
    stroke_guides: float = 0.004, # guides stroke = UPEM * stroke_guides
    margin_em = 0.01,             # margins as a fraction of UPEM (float or [top,right,bottom,left])
    bitmap_style: str = "image",  # "image", "rects", "circles"
    bitmap_opacity: float = 1,  # opacity for bitmap layer (0.0 – 1.0)
    bitmap_scale: float = 1,  # scale factor for bitmap pixels (0.9 makes them 90% size, centered)
):
    """
    Generate an SVG visualization showing how font hinting affects glyph rendering.
    
    Creates a scalable vector graphic in EM coordinate space that overlays:
    - Original (unhinted) glyph outline in font design units
    - Hinted outline converted back to EM units for comparison  
    - Bitmap renderings (from hinted or unhinted outlines) scaled to EM space
    - Font metrics guides and labels for analysis
    
    This allows direct visual comparison of how hinting algorithms modify the 
    original glyph design to improve rendering at small pixel sizes.

    Args:
        font_path: Path to the font file (TTF, OTF, etc.)
        char: Single character to analyze
        ppem: Pixels per EM - the target rendering size for hinting
        hinting_target: Hinting algorithm ("normal", "light", "mono", "lcd", "lcd_v")
        use_hinted_bitmap: Whether bitmap layer uses hinted (True) or unhinted (False) outline
        layers: Comma-separated list of visualization layers:
                - path-original: Original unhinted outline (solid black)
                - path-hinted: Hinted outline (dashed blue)  
                - bitmap-original: Bitmap from unhinted outline
                - bitmap-hinted: Bitmap from hinted outline
                - guides: Ascender/descender lines (gold)
                - baseline: Baseline reference line (gray)
                - bearings: Left side bearing and advance lines (blue/green)
                - bbox: Glyph bounding box (red)
                - labels: Text annotations and legend
        output_filename: Custom filename for output SVG file. If None, auto-generates 
                        filename based on character, hinting target, and layers
        label_scale: Size of text labels as fraction of UPEM
        stroke_main: Outline stroke width as fraction of UPEM
        stroke_guides: Guide line stroke width as fraction of UPEM  
        margin_em: SVG margins as fraction of UPEM (float for all sides, or [top,right,bottom,left] array)
        bitmap_style: How to render bitmap pixels ("image", "rects", "circles")
        bitmap_opacity: Transparency of bitmap overlay (0.0-1.0)
        bitmap_scale: Scale factor for bitmap pixels (1.0 = full size, 0.9 = 90% size, centered)

    Returns:
        Dictionary with output file path, metrics, and generation details
        
    The output SVG uses EM units throughout, making it resolution-independent
    while clearly showing the pixel-level effects of hinting at the target PPEM.
    """
    # Load the font face
    face = ft.Face(font_path)
    
    # Set variable font coordinates if this is a variable font
    # Weight: Regular 400, Width: Normal 100
    try:
        # Get the variable axes information
        var_info = face.get_variation_info()
        if var_info and len(var_info.axes) > 0:
            # Set coordinates for Weight=400 and Width=100
            coords = []
            axes_info = []
            
            for axis in var_info.axes:
                tag = axis.tag if isinstance(axis.tag, str) else axis.tag.decode()
                if tag == 'wght':  # Weight axis
                    coords.append(400.0)
                    axes_info.append(f'{tag}: 400.0')
                elif tag == 'wdth':  # Width axis  
                    coords.append(100.0)
                    axes_info.append(f'{tag}: 100.0')
                else:
                    # Use default value for other axes
                    coords.append(axis.default)
                    axes_info.append(f'{tag}: {axis.default}')
            
            if coords:
                face.set_var_design_coords(coords)
                print(f"Set variable font coordinates: [{', '.join(axes_info)}]")
    except Exception as e:
        print(f"Note: Could not set variable font coordinates: {e}")
        print("Proceeding with default font instance...")
    
    # Get the font's units per EM (typically 1000, 2048, or 4096)
    # This is the resolution of the original font design grid
    upem = face.units_per_EM

    # Parse the comma-separated layers list into a set for fast lookup
    layer_set = set(layer.strip() for layer in layers.split(",") if layer.strip())
    
    # Determine which bitmap renderings we need to generate based on requested layers
    # This optimization avoids unnecessary bitmap generation for better performance
    need_hinted_bitmap = "bitmap-hinted" in layer_set
    need_unhinted_bitmap = "bitmap-original" in layer_set
    need_any_bitmap = need_hinted_bitmap or need_unhinted_bitmap

    # =================================================================
    # STEP 1: Load original (unhinted, unscaled) glyph data
    # =================================================================
    # Load the glyph without hinting or scaling to get the original design
    # This gives us the glyph as the font designer intended it
    face.load_char(char, ft.FT_LOAD_NO_HINTING | ft.FT_LOAD_NO_SCALE)
    orig_slot = face.glyph
    orig_outline = orig_slot.outline
    om = orig_slot.metrics  # All values are in font design units (EM units)

    # Extract glyph metrics in EM units (no scaling factor needed with NO_SCALE)
    lsb_em    = float(om.horiBearingX)  # Left side bearing
    top_em    = float(om.horiBearingY)  # Distance from baseline to top of glyph
    width_em  = float(om.width)         # Glyph width
    height_em = float(om.height)        # Glyph height  
    adv_em    = float(om.horiAdvance)   # Horizontal advance (spacing to next glyph)
    
    # Font-wide vertical metrics in EM units
    asc_em    = float(face.ascender)    # Ascender height (positive)
    desc_em   = float(face.descender)   # Descender depth (negative)

    # Helper function to convert FreeType outline to SVG path data
    # Handles coordinate system conversion: FreeType uses Y-up, SVG uses Y-down
    def em_path_from_outline(outline):
        """Convert FreeType outline to SVG path string in EM coordinates."""
        cmds, open_flag = [], False
        
        def move_to(p, _):
            nonlocal open_flag
            if open_flag: cmds.append("Z")  # Close previous path
            cmds.append(f"M {p.x:.3f} {-p.y:.3f}")  # Flip Y coordinate
            open_flag = True
            
        def line_to(p, _):
            cmds.append(f"L {p.x:.3f} {-p.y:.3f}")
            
        def conic_to(c, p, _):
            # Quadratic Bézier curve (TrueType style)
            cmds.append(f"Q {c.x:.3f} {-c.y:.3f} {p.x:.3f} {-p.y:.3f}")
            
        def cubic_to(c1, c2, p, _):
            # Cubic Bézier curve (PostScript/OpenType style)
            cmds.append(
                "C "
                f"{c1.x:.3f} {-c1.y:.3f} "
                f"{c2.x:.3f} {-c2.y:.3f} "
                f"{p.x:.3f} {-p.y:.3f}"
            )
            
        # Decompose the outline into path commands
        outline.decompose(move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
        if open_flag: cmds.append("Z")  # Close final path
        return " ".join(cmds)

    orig_path_em = em_path_from_outline(orig_outline)

    # =================================================================
    # STEP 2: Load hinted glyph data and convert back to EM units
    # =================================================================
    # Set the target pixel size - this is where hinting becomes active
    face.set_pixel_sizes(ppem, 0)
    
    # Load the glyph with hinting enabled to see how it's modified for pixel rendering
    # The outline coordinates will be in 26.6 fixed-point pixel units
    flags_hinted = ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
    face.load_char(char, flags_hinted)
    hinted_slot = face.glyph
    hm = hinted_slot.metrics  # Metrics in 26.6 fixed-point pixel units
    
    # Get the actual pixels-per-EM that FreeType is using (may differ slightly from requested)
    x_ppem = face.size.x_ppem
    y_ppem = face.size.y_ppem

    # Conversion functions: 26.6 fixed-point pixels → EM units
    # This allows us to overlay hinted outlines on the original EM-space design
    def px_to_em_x(px26_6): 
        return (px26_6 / 64.0) * (upem / float(x_ppem))
    def px_to_em_y(px26_6): 
        return (px26_6 / 64.0) * (upem / float(y_ppem))

    def hinted_em_path(outline):
        """Convert hinted outline from pixel coordinates back to EM coordinates."""
        cmds, open_flag = [], False
        
        def move_to(p, _):
            nonlocal open_flag
            if open_flag: cmds.append("Z")
            # Convert pixel coordinates to EM and flip Y
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

    # =================================================================
    # STEP 3: Generate bitmap renderings and convert to EM coordinates
    # =================================================================
    render_mode = _RENDER_MODE[hinting_target]
    bitmaps = {}  # Store bitmap data for each type we generate
    
    if need_hinted_bitmap:
        # Generate bitmap from the hinted outline (already loaded above)
        hinted_slot.render(render_mode)
        hinted_bmp = hinted_slot.bitmap
        hinted_bmp_left_px = hinted_slot.bitmap_left  # Bitmap position relative to origin
        hinted_bmp_top_px  = hinted_slot.bitmap_top
        
        # For proper alignment, position bitmap directly using FreeType's calculated position
        # bitmap_left and bitmap_top already give us the correct position relative to the baseline and origin
        # No additional offset calculation needed - FreeType has already positioned it correctly
        
        # Store all bitmap data converted to EM coordinates
        bitmaps['hinted'] = {
            'bitmap': hinted_bmp,
            'img_x_em': hinted_bmp_left_px * (upem / float(x_ppem)),   # Bitmap left edge (already correctly positioned)
            'img_y_em': -hinted_bmp_top_px * (upem / float(y_ppem)),   # Bitmap top edge (Y-flipped, baseline-relative)
            'img_w_em': hinted_bmp.width * (upem / float(x_ppem)),     # Bitmap width
            'img_h_em': hinted_bmp.rows * (upem / float(y_ppem)),      # Bitmap height
            'dx_em': 0.0,  # No additional offset needed - FreeType positioning is correct
            'dy_em': 0.0,  # No additional offset needed - FreeType positioning is correct
            'png_uri': _png_data_uri_from_bitmap(hinted_bmp)  # Base64 PNG for SVG embedding
        }
    
    if need_unhinted_bitmap:
        # Generate bitmap from unhinted outline at the same pixel size for comparison
        flags_unhinted = ft.FT_LOAD_NO_HINTING | ft.FT_LOAD_NO_BITMAP | _LOAD_TARGET[hinting_target]
        face.load_char(char, flags_unhinted)
        unhinted_slot = face.glyph
        unhinted_slot.render(render_mode)
        unhinted_bmp = unhinted_slot.bitmap
        unhinted_bmp_left_px = unhinted_slot.bitmap_left
        unhinted_bmp_top_px  = unhinted_slot.bitmap_top
        
        # For unhinted bitmap, also use FreeType's positioning directly
        # This ensures consistent baseline and origin alignment with the hinted version
        
        bitmaps['unhinted'] = {
            'bitmap': unhinted_bmp,
            'img_x_em': unhinted_bmp_left_px * (upem / float(x_ppem)),
            'img_y_em': -unhinted_bmp_top_px * (upem / float(y_ppem)),
            'img_w_em': unhinted_bmp.width * (upem / float(x_ppem)),
            'img_h_em': unhinted_bmp.rows * (upem / float(y_ppem)),
            'dx_em': 0.0,  # No additional offset needed
            'dy_em': 0.0,  # No additional offset needed
            'png_uri': _png_data_uri_from_bitmap(unhinted_bmp)
        }

    # =================================================================
    # STEP 4: Calculate SVG viewBox and styling
    # =================================================================
    
    # Determine the overall bounds for the SVG viewBox
    # Include bitmap extents if any bitmaps were generated
    if bitmaps:
        primary_bitmap = list(bitmaps.values())[0]
        img_left_em = primary_bitmap['img_x_em'] + primary_bitmap['dx_em']
        img_right_em = img_left_em + primary_bitmap['img_w_em']
    else:
        # No bitmaps requested, use zero extents
        img_left_em = img_right_em = 0.0

    # Parse margin_em parameter - supports CSS-like margin syntax:
    # - Single float: same margin for all sides
    # - [top, right, bottom, left]: individual margins for each side
    # - [vertical, horizontal]: top/bottom and left/right margins
    # - [value]: single value in array format
    # Values can be negative to crop into content area
    if isinstance(margin_em, (list, tuple)):
        if len(margin_em) == 4:
            margin_top, margin_right, margin_bottom, margin_left = margin_em
        elif len(margin_em) == 2:
            # [vertical, horizontal] format
            margin_top = margin_bottom = margin_em[0]
            margin_right = margin_left = margin_em[1]
        elif len(margin_em) == 1:
            # Single value in array
            margin_top = margin_right = margin_bottom = margin_left = margin_em[0]
        else:
            raise ValueError("margin_em array must have 1, 2, or 4 values")
    else:
        # Single value for all sides
        margin_top = margin_right = margin_bottom = margin_left = margin_em
    
    # Convert margins to EM units
    margin_top_em = margin_top * upem
    margin_right_em = margin_right * upem
    margin_bottom_em = margin_bottom * upem
    margin_left_em = margin_left * upem
    
    # Calculate SVG viewBox in EM units with individual margins
    left  = min(0.0, lsb_em, img_left_em) - margin_left_em     # Leftmost extent
    right = max(adv_em, lsb_em + width_em, img_right_em) + margin_right_em  # Rightmost extent
    top_vb    = -(asc_em + margin_top_em)                       # Top of viewBox (SVG Y-down)
    height_vb = (asc_em - desc_em) + margin_top_em + margin_bottom_em  # Total height
    width_vb  = right - left                                    # Total width

    # Scale all visual elements proportionally to UPEM for resolution independence
    fs = label_scale * upem        # Font size for labels
    sw_main = stroke_main * upem   # Stroke width for main outlines
    sw_guid = stroke_guides * upem # Stroke width for guide lines
    dash_a = 0.0 * upem           # Dash pattern (solid)
    dash_b = 0.007 * upem         # Dash pattern (gap)

    # Define visual styles for different path types
    path_original_style = f'stroke="#000" stroke-width="{sw_main:.3f}"'  # Black solid
    path_hinted_style = f'stroke="#0054a2" stroke-width="{sw_main*1.5:.3f}" stroke-dasharray="{dash_a:.3f} {dash_b:.3f}" stroke-linecap="round"'  # Blue dashed

    

    def label(x, y, text, anchor="start"):
        """Generate SVG text element with consistent styling."""
        return (f'<text x="{x:.3f}" y="{y:.3f}" font-size="{fs:.3f}" '
                f'text-anchor="{anchor}" font-family="monospace">{text}</text>')

    # =================================================================
    # STEP 5: Generate SVG markup
    # =================================================================
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{left:.3f} {top_vb:.3f} {width_vb:.3f} {height_vb:.3f}" '
        f'width="{width_vb:.0f}" height="{height_vb:.0f}">'
    )
    svg.append(f'<!-- Freetype version: {version} -->')
    svg.append('<rect x="{:.3f}" y="{:.3f}" width="{:.3f}" height="{:.3f}" fill="#fff"/>'
               .format(left, top_vb, width_vb, height_vb))

    # Helper function to render bitmap overlay in the requested style
    def render_bitmap(bitmap_info, bitmap_type):
        bmp = bitmap_info['bitmap']
        img_x_em = bitmap_info['img_x_em']
        img_y_em = bitmap_info['img_y_em']
        img_w_em = bitmap_info['img_w_em']
        img_h_em = bitmap_info['img_h_em']
        png_uri = bitmap_info['png_uri']
        
        if bitmap_style == "image":
            # Embed the bitmap as a PNG image with pixelated rendering
            opacity_style = f" opacity: {bitmap_opacity:.3f};" if bitmap_opacity < 1.0 else ""
            svg.append(
                f'<image x="{img_x_em:.3f}" y="{img_y_em:.3f}" '
                f'width="{img_w_em:.3f}" height="{img_h_em:.3f}" '
                f'href="{png_uri}" '
                f'style="image-rendering: pixelated;{opacity_style}"/>'
            )
        else:
            # Render individual pixels as SVG shapes (rects or circles)
            # Process bitmap pixel by pixel to create individual SVG elements
            alpha = bmp.buffer
            w, h, pitch = bmp.width, bmp.rows, bmp.pitch
            px_w = upem / float(x_ppem)  # Pixel width in EM units
            px_h = upem / float(y_ppem)  # Pixel height in EM units
            
            for y in range(h):
                # Pre-extract row for grayscale bitmaps (optimization)
                row = alpha[y * pitch : y * pitch + w] if bmp.pixel_mode == ft.FT_PIXEL_MODE_GRAY else None
                
                for x in range(w):
                    # Extract pixel value based on bitmap format
                    if bmp.pixel_mode == ft.FT_PIXEL_MODE_GRAY:
                        a = row[x]  # 8-bit grayscale value
                    elif bmp.pixel_mode == ft.FT_PIXEL_MODE_MONO:
                        # Extract bit from packed monochrome data
                        b = alpha[y * pitch + (x >> 3)]
                        a = 255 if (b >> (7 - (x & 7))) & 1 else 0
                    else:
                        a = 0  # Unsupported format
                        
                    if a == 0:
                        continue  # Skip transparent pixels
                        
                    # Calculate pixel position in EM coordinates (no additional offset needed)
                    cx = img_x_em + x * px_w
                    cy = img_y_em + y * px_h
                    
                    # Apply scaling factor and center the scaled shape within the pixel
                    scaled_w = px_w * bitmap_scale
                    scaled_h = px_h * bitmap_scale
                    offset_x = (px_w - scaled_w) * 0.5  # Center horizontally
                    offset_y = (px_h - scaled_h) * 0.5  # Center vertically
                    
                    # Convert alpha to grayscale color (higher alpha = darker)
                    intensity = a / 255.0
                    gray_value = int(255 * (1.0 - intensity))
                    fill_color = f"rgb({gray_value},{gray_value},{gray_value})"
                    
                    opacity_attr = f' fill-opacity="{bitmap_opacity:.3f}"' if bitmap_opacity < 1.0 else ""
                    
                    if bitmap_style == "rects":
                        # Rectangular pixels (most accurate representation)
                        rect_x = cx + offset_x
                        rect_y = cy + offset_y
                        svg.append(
                            f'<rect x="{rect_x:.3f}" y="{rect_y:.3f}" width="{scaled_w:.3f}" height="{scaled_h:.3f}" '
                            f'fill="{fill_color}"{opacity_attr}/>'
                        )
                    elif bitmap_style == "circles":
                        # Circular pixels (softer appearance)
                        r = min(scaled_w, scaled_h) * 0.5  # Use smaller dimension for radius to maintain aspect
                        circle_cx = cx + px_w * 0.5  # Center of original pixel
                        circle_cy = cy + px_h * 0.5
                        svg.append(
                            f'<circle cx="{circle_cx:.3f}" cy="{circle_cy:.3f}" r="{r:.3f}" '
                            f'fill="{fill_color}"{opacity_attr}/>'
                        )

    # =================================================================
    # Layer rendering order (bottom to top):
    # 1. Bitmap layers (background)
    # 2. Guide lines 
    # 3. Vector outlines (foreground)
    # 4. Labels and legend (top)
    # =================================================================
    
    # Bitmap layers (rendered first, appear behind other elements)
    if "bitmap-original" in layer_set and "unhinted" in bitmaps:
        render_bitmap(bitmaps["unhinted"], "unhinted")
    
    if "bitmap-hinted" in layer_set and "hinted" in bitmaps:
        render_bitmap(bitmaps["hinted"], "hinted")

    # Font metric guide lines
    if "guides" in layer_set:

        # Ascender line (maximum height for tall letters like 'h', 'k')
        svg.append(
            f'<line x1="{left:.3f}" y1="{-asc_em:.3f}" x2="{right:.3f}" y2="{-asc_em:.3f}" '
            f'stroke="#e0a000" stroke-width="{sw_guid:.3f}"/>'
        )
        # Descender line (minimum depth for letters like 'g', 'p')
        svg.append(
            f'<line x1="{left:.3f}" y1="{-desc_em:.3f}" x2="{right:.3f}" y2="{-desc_em:.3f}" '
            f'stroke="#e0a000" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(left + 0.02 * upem, -asc_em - 0.02 * upem, "ascender"))
            svg.append(label(left + 0.02 * upem, -desc_em + 0.06 * upem, "descender"))

    # Baseline (Y=0, where most letters sit)
    if "baseline" in layer_set:
        svg.append(
            f'<line x1="{left:.3f}" y1="{0:.3f}" x2="{right:.3f}" y2="{0:.3f}" '
            f'stroke="#c0c0c0" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(left + 0.02 * upem, -0.02 * upem, "baseline"))

    # Spacing and positioning guides (from original EM metrics)
    if "bearings" in layer_set:
        # Origin line (X=0, glyph positioning reference)
        svg.append(
            f'<line x1="0" y1="{-asc_em:.3f}" x2="0" y2="{-desc_em:.3f}" '
            f'stroke="#c0c0c0" stroke-width="{sw_guid:.3f}"/>'
        )
        # Left side bearing line (where glyph content begins)
        svg.append(
            f'<line x1="{lsb_em:.3f}" y1="{-asc_em:.3f}" x2="{lsb_em:.3f}" y2="{-desc_em:.3f}" '
            f'stroke="#00a0e0" stroke-width="{sw_guid:.3f}"/>'
        )
        # Advance width line (where next glyph would start)
        svg.append(
            f'<line x1="{adv_em:.3f}" y1="{-asc_em:.3f}" x2="{adv_em:.3f}" y2="{-desc_em:.3f}" '
            f'stroke="#008000" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(0, -asc_em - 0.04 * upem, "origin x=0", anchor="middle"))
            svg.append(label(lsb_em, -asc_em - 0.04 * upem, f"LSB={lsb_em:.1f} em-units", anchor="middle"))
            svg.append(label(adv_em, -asc_em - 0.04 * upem, f"advance={adv_em:.1f} em-units", anchor="middle"))

    # Glyph bounding box (tightest rectangle around the glyph)
    if "bbox" in layer_set:
        svg.append(
            f'<rect x="{lsb_em:.3f}" y="{-top_em:.3f}" width="{width_em:.3f}" height="{height_em:.3f}" '
            f'fill="none" stroke="#cc3333" stroke-width="{sw_guid:.3f}"/>'
        )
        if "labels" in layer_set:
            svg.append(label(lsb_em + 0.02 * upem, -top_em + 0.06 * upem,
                             f"bbox {width_em:.0f}×{height_em:.0f} em-units"))

    # Vector outline paths (main content)
    if "path-original" in layer_set:
        # Original unhinted outline as designed by the font creator
        svg.append(
            f'<path d="{orig_path_em}" fill="none" {path_original_style} />'
        )
        
    if "path-hinted" in layer_set:
        # Hinted outline converted back to EM units for comparison
        svg.append(
            f'<path d="{hinted_path_em}" fill="none" {path_hinted_style} />'
        )

    # Visual legend explaining the different elements
    if "labels" in layer_set:
        lx = left + 0.04 * upem
        ly = -asc_em + 0.10 * upem
        
        if "path-original" in layer_set:
            svg.append(label(lx + 0.1 * upem, ly, "original outline", anchor="start"))
            svg.append(
                f'<line x1="{lx:.3f}" y1="{ly - 0.02 * upem:.3f}" '
                f'x2="{lx + 0.05 * upem:.3f}" y2="{ly - 0.02 * upem:.3f}" '
                f'{path_original_style} />'
            )
            ly += 0.08 * upem
            
        if "path-hinted" in layer_set:
            svg.append(label(lx + 0.1 * upem, ly, "hinted outline", anchor="start"))
            svg.append(
                f'<line x1="{lx:.3f}" y1="{ly - 0.02 * upem:.3f}" '
                f'x2="{lx + 0.05 * upem:.3f}" y2="{ly - 0.02 * upem:.3f}" '
                f'{path_hinted_style} />'
            )
            ly += 0.08 * upem
            
        if "bitmap-hinted" in layer_set:
            svg.append(label(lx + 0.1 * upem, ly, "bitmap hinted", anchor="start"))
            svg.append(
                f'<rect x="{lx:.3f}" y="{ly - 0.04 * upem:.3f}" '
                f'width="{0.05 * upem:.3f}" height="{0.05 * upem:.3f}" '
                f'fill="#000" fill-opacity="0.4" stroke="none"/>'
            )
            ly += 0.08 * upem
            
        if "bitmap-original" in layer_set:
            svg.append(label(lx + 0.1 * upem, ly, "bitmap unhinted", anchor="start"))
            svg.append(
                f'<rect x="{lx:.3f}" y="{ly - 0.04 * upem:.3f}" '
                f'width="{0.05 * upem:.3f}" height="{0.05 * upem:.3f}" '
                f'fill="#000" fill-opacity="0.6" stroke="none"/>'
            )

    svg.append("</svg>")
    
    # =================================================================
    # STEP 6: Save output file with descriptive name
    # =================================================================
    # Use custom filename if provided, otherwise generate based on character, hinting type, and active layers
    if output_filename is not None:
        out_svg = output_filename
    else:
        layer_suffix = "_".join(sorted(layer_set)) if layer_set else "empty"
        layer_suffix = layer_suffix.replace("-", "").replace(",", "_")  # Clean up for filesystem
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
    """
    Example usage demonstrating different visualization modes.
    Each example shows different aspects of font hinting effects.
    """

    # font path
    font_path = "ARIALUNI.TTF"
    # font_path = "OpenSans-VariableFont_wdth,wght.ttf"
    # font_path = "WorkSans-VariableFont_wght.ttf"
    # font_path = "Roboto[ital,wdth,wght].ttf"

    # font size
    ppem = 12

    # character
    char = "a"

    # let's crop the output svg to the glyph only (remove some whitespace)
    margin_em=[-.45, 0, -0.15, 0]  # [top, right, bottom, left]
    
    # Outline only
    info = glyph_em_overlay(
        font_path, char,
        output_filename="./docs/asset/path-original.svg",
        ppem=ppem, hinting_target="mono",
        layers="path-original",
        margin_em=margin_em,
    )
    print("Wrote file:", info["out_svg"])

    # Outline + bitmap (mono)
    info = glyph_em_overlay(
        font_path, char,
        output_filename="./docs/asset/path-original_mono-bitmap-original.svg",
        ppem=ppem, hinting_target="mono",
        layers="path-original,bitmap-original",
        bitmap_style="rects",
        bitmap_opacity=0.25,
        margin_em=margin_em,
    )
    print("Wrote file:", info["out_svg"])
    
    # Comparison of original outline, hinted outline
    info = glyph_em_overlay(
        font_path, char, 
        output_filename="./docs/asset/path-original_path-hinted.svg",
        ppem=ppem, hinting_target="mono",
        layers="path-original,path-hinted",
        bitmap_opacity=0.25,
        margin_em=margin_em,
    )
    print("Wrote file:", info["out_svg"])

    # Outline hinted + bitmap (mono)
    info = glyph_em_overlay(
        font_path, char, 
        output_filename="./docs/asset/path-hinted_mono-bitmap-hinted.svg",
        ppem=ppem, hinting_target="mono",
        layers="path-original,path-hinted,bitmap-hinted",
        bitmap_style="rects",
        bitmap_opacity=0.25,
        margin_em=margin_em,
    )
    print("Wrote file:", info["out_svg"])

    # Outline + bitmap (normal)
    info = glyph_em_overlay(
        font_path, char, 
        output_filename="./docs/asset/path-original_8bit-bitmap-original.svg",
        ppem=ppem, hinting_target="normal",
        layers="path-original,bitmap-original",
        bitmap_style="rects",
        margin_em=margin_em,
    )
    print("Wrote file:", info["out_svg"])

    # Outline + bitmap (normal)
    info = glyph_em_overlay(
        font_path, char, 
        output_filename="./docs/asset/path-hinted_8bit-bitmap-hinted.svg",
        ppem=ppem, hinting_target="normal",
        layers="path-hinted,bitmap-hinted",
        margin_em=margin_em,
        bitmap_style="rects",
    )
    print("Wrote  file:", info["out_svg"])

    # Complete analysis with all guides and metrics
    info = glyph_em_overlay(
        font_path, char, 
        output_filename="./docs/asset/debug.svg",
        ppem=ppem, hinting_target="normal",
        layers="path-original,path-hinted,bitmap-hinted,guides,baseline,bearings,bbox,labels",
        bitmap_style="rects",
    )
    print("Wrote file:", info["out_svg"])
