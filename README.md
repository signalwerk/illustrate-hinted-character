# Font Hinting Visualization Tool

See [here](https://hinting.signalwerk.ch/) for the live version.

This program creates SVG visualizations that demonstrate how font hinting affects glyph outlines and their bitmap rendering. It helps illustrate the difference between the original font design and how hinting algorithms modify it for better pixel-grid alignment at small sizes.

## What it shows

- **Original outline**: The glyph as designed by the font creator (unhinted)
- **Hinted outline**: How the hinting algorithm modifies the outline for pixel rendering
- **Bitmap renderings**: The actual pixels generated from both hinted and unhinted outlines
- **Font metrics**: Guidelines showing ascender, descender, baseline, bearings, and bounding boxes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This will generate several example SVG files showing different visualization modes for the letter 'a' using the Arial Unicode font.

## Understanding Font Hinting

Font hinting is a process that adjusts glyph outlines to align better with the pixel grid at small sizes. Without hinting, curved lines and diagonal strokes can appear blurry or inconsistent when rendered at low resolutions. The visualization shows:

1. **How outlines change**: Compare the original design with the hinted version
2. **Pixel-level effects**: See exactly which pixels get filled in each approach
3. **Metrics preservation**: Observe how spacing and positioning are maintained

The SVG output uses EM coordinates throughout, making it resolution-independent while clearly showing the pixel-level effects of hinting at the target size.
