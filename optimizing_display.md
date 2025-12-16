# Optimizing Images for Spectra 6 E-ink Display

This document explains the image processing parameters in `send_to_display.py` and how they affect the final output on the e-ink display.

## Processing Pipeline

Images go through two distinct stages before being sent to the display:

```
Input Image
    ↓
┌─────────────────────────────┐
│ Stage 1: Image Enhancement  │  ← Brightness, Contrast, Saturation
│ (modifies actual pixels)    │
└─────────────────────────────┘
    ↓
Enhanced Image (still millions of colors)
    ↓
┌─────────────────────────────┐
│ Stage 2: Color Matching     │  ← Luminance weight, Chroma weight
│ (picks from 6 palette colors)│
└─────────────────────────────┘
    ↓
Dithered Output (6 colors only)
```

## Parameter Reference

### Stage 1: Image Enhancement

These parameters modify the actual pixel values of the input image before any color matching occurs.

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Brightness | `-b` | 1.0 | Makes input lighter (>1) or darker (<1) |
| Contrast | `-c` | 1.2 | Spreads (>1) or compresses (<1) tonal range |
| Saturation | `-s` | 1.2 | Makes colors more vivid (>1) or muted (<1, 0=grayscale) |

### Stage 2: Color Matching (Lab Color Space)

These parameters control how the Floyd-Steinberg dithering algorithm decides which of the 6 palette colors is "closest" to each pixel. The algorithm uses CIE Lab color space, which is perceptually uniform (distances correspond to how humans perceive color differences).

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Luminance weight | `-l` | 0.2 | How much brightness difference matters (range: 0.0-2.0) |
| Chroma weight | `-C` | 3.0 | How much color/hue difference matters (range: 0.5-5.0) |

The distance formula is:
```
distance = sqrt(luminance_weight × ΔL² + chroma_weight × Δa² + chroma_weight × Δb²)
```

Where L is lightness, and a/b are the color components in Lab space.

## Why Lab Color Space?

The original implementation used PIL's built-in quantization, which matches colors using RGB Euclidean distance. RGB distance is not perceptually uniform—colors that appear very different to humans may have similar RGB distances.

Lab color space was designed to match human vision:
- Equal distances in Lab space correspond to equal perceived differences
- The L channel (luminance) is separate from color (a and b channels)
- This allows independent control over brightness vs. color matching

## Understanding the Interaction

### Saturation vs. Chroma Weight

These sound similar but operate differently:

| Parameter | What it does | Example |
|-----------|--------------|---------|
| **Saturation 1.5** | Changes input pixels: muted pink (200,150,150) becomes vivid pink (220,110,110) | Affects the image before matching |
| **Chroma weight 5.0** | Changes matching: pink is matched to Red instead of White | Affects algorithm decisions |

### Contrast vs. Luminance Weight

| Parameter | What it does | Example |
|-----------|--------------|---------|
| **Contrast 1.5** | Spreads tonal range: mid-grays become lighter/darker | More distinct light/dark areas |
| **Luminance weight 2.0** | Matching prefers brightness accuracy over color | Dark blue may become black |

## Recommended Settings for E-ink

E-ink displays have inherent limitations: muted colors, limited dynamic range, and only 6 available colors. The defaults are tuned for this:

### Conservative Starting Point
```bash
uv run send_to_display.py image.jpg --show
# Uses: -c 1.2 -s 1.2 -l 0.2 -C 3.0
```

### Test Configurations

| Goal | Command |
|------|---------|
| Standard CIE76 (baseline) | `-l 1.0 -C 1.0` |
| Strong color preference | `-l 0.1 -C 3.0` |
| Ignore brightness entirely | `-l 0.0 -C 1.0` |
| Extreme color emphasis | `-l 0.1 -C 5.0` |
| Tonal/brightness emphasis | `-l 2.0 -C 0.5` |

### When to Adjust Each Parameter

| Symptom | Try adjusting |
|---------|---------------|
| Colors look washed out | ↑ Saturation or ↑ Chroma weight |
| Image looks flat/gray | ↑ Contrast |
| Dark colors becoming black | ↓ Luminance weight |
| Colors shifting unexpectedly | ↑ Luminance weight |
| Too much dithering noise | Try different chroma/luminance balance |

## Best Practices

1. **Always preview first**: Use `--show` to see the dithered result before sending to the display

2. **Test with representative images**: Use photos that have the colors you care about (saturated colors, skin tones, etc.)

3. **Start with defaults**: The defaults (contrast 1.2, saturation 1.2, luminance 0.2, chroma 3.0) are reasonable for most images

4. **Adjust one parameter at a time**: This helps you understand what each change does

5. **Consider the source**: Dark/underexposed photos may need brightness adjustment; already-vivid photos may need less saturation boost

## The 6-Color Palette

The Spectra 6 display can show these colors:

| Color | RGB Value | Hardware Code |
|-------|-----------|---------------|
| Black | (0, 0, 0) | 0x00 |
| White | (255, 255, 255) | 0x01 |
| Yellow | (255, 255, 0) | 0x02 |
| Red | (255, 0, 0) | 0x03 |
| Blue | (0, 0, 255) | 0x05 |
| Green | (41, 204, 20) | 0x06 |

Note: These are idealized RGB values. The actual colors produced by the e-ink pigments are more muted. For optimal results, you could measure your display's actual colors and update `PALETTE_RGB` in the script.
