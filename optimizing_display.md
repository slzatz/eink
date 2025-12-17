# Optimizing Images for Spectra 6 E-ink Display

This document explains the image processing parameters in `send_to_display.py` and how they affect the final output on the e-ink display.

## Processing Pipeline

Images go through these stages before being sent to the display:

```
Input Image
    ↓
┌─────────────────────────────┐
│ 1. Resize/Crop              │  Fit to 1200×1600 (LANCZOS)
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ 2. Image Enhancement        │  ← Brightness, Contrast, Saturation
│ (modifies actual pixels)    │
└─────────────────────────────┘
    ↓
Enhanced Image (still millions of colors)
    ↓
┌─────────────────────────────┐
│ 3. Floyd-Steinberg Dither   │  PIL's built-in quantization
│ (reduces to 6 palette colors)│
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ 4. Pack & Upload            │  2 pixels per byte → ESP32
└─────────────────────────────┘
```

## Parameter Reference

### Image Enhancement Parameters

These parameters modify the actual pixel values of the input image before dithering.

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Brightness | `-b` | 1.0 | Makes image lighter (>1) or darker (<1) |
| Contrast | `-c` | 1.2 | Spreads (>1) or compresses (<1) tonal range |
| Saturation | `-s` | 1.2 | Makes colors more vivid (>1) or muted (<1, 0=grayscale) |

### Preview Option

| Option | Flag | Description |
|--------|------|-------------|
| Show preview | `--show` | Save preview to `/tmp/spectra_preview.png` and open with ImageMagick |

## Understanding Enhancement Parameters

### Contrast
Controls the spread between light and dark areas:
- **>1.0**: More dramatic light/dark separation, punchier image
- **<1.0**: Flatter, more compressed tonal range
- **Default 1.2**: Slight boost helps compensate for e-ink's limited dynamic range

### Brightness
Shifts the overall lightness:
- **>1.0**: Brighter image (good for underexposed photos)
- **<1.0**: Darker image
- **Default 1.0**: No change

### Saturation
Controls color intensity:
- **>1.0**: More vivid, saturated colors
- **<1.0**: More muted colors
- **0.0**: Grayscale
- **Default 1.2**: Slight boost helps colors pop on the muted e-ink display

## Recommended Settings

### Default (good starting point)
```bash
uv run send_to_display.py image.jpg --show
# Uses: -c 1.2 -b 1.0 -s 1.2
```

### Test Configurations

| Goal | Command |
|------|---------|
| No enhancement (original) | `-c 1.0 -b 1.0 -s 1.0` |
| High contrast | `-c 1.5` |
| Vivid colors | `-s 1.5` |
| Bright + colorful | `-b 1.1 -s 1.4` |
| Dramatic | `-c 1.4 -s 1.3` |
| Muted/artistic | `-c 0.9 -s 0.8` |

### When to Adjust Each Parameter

| Symptom | Try adjusting |
|---------|---------------|
| Colors look washed out | ↑ Saturation (`-s 1.4` or higher) |
| Image looks flat/gray | ↑ Contrast (`-c 1.3` or higher) |
| Image too dark | ↑ Brightness (`-b 1.1` or higher) |
| Image too bright/blown out | ↓ Brightness (`-b 0.9`) |
| Colors too intense/garish | ↓ Saturation (`-s 1.0`) |

## Best Practices

1. **Always preview first**: Use `--show` to see the dithered result before sending to the display

2. **Test with representative images**: Use photos that have the colors you care about

3. **Start with defaults**: The defaults (contrast 1.2, saturation 1.2) work well for most images

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

## Technical Notes

### Why PIL's Built-in Dithering?

An earlier version of this script used custom Floyd-Steinberg dithering with CIE Lab color space for perceptually-accurate color matching. Testing revealed that while Lab-based matching changed ~43% of pixel assignments compared to PIL's RGB-based approach, the visual results were essentially identical due to how dithering error diffusion works.

PIL's built-in dithering offers:
- **~24x faster processing** (~1.4 seconds vs ~34 seconds)
- **Simpler code** (removed numpy dependency)
- **Equivalent visual quality**

The enhancement parameters (contrast, brightness, saturation) have a much more significant impact on the final image than the color-matching algorithm used during dithering.
