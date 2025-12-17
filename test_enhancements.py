#!/usr/bin/env python3
"""
Diagnostic tool to compare how contrast, brightness, and saturation
parameters affect the dithered output for the Spectra 6 display.

This script helps visualize and quantify how image enhancement settings
change the final dithered result.
"""

import sys
import argparse
import math
from collections import defaultdict
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# --- CONFIGURATION ---
FRAME_WIDTH = 1200
FRAME_HEIGHT = 1600

# The Spectra 6 Color Palette (RGB)
PALETTE_RGB = [
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (255, 255, 0),   # Yellow
    (255, 0, 0),     # Red
    (0, 0, 255),     # Blue
    (41, 204, 20)    # Green
]

COLOR_NAMES = ["Black", "White", "Yellow", "Red", "Blue", "Green"]

# --- LAB COLOR SPACE FUNCTIONS ---

def rgb_to_lab(r, g, b):
    """Convert RGB (0-255) to CIE Lab color space."""
    # Normalize RGB to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Apply gamma correction (sRGB to linear)
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92

    # Scale to 100
    r, g, b = r * 100, g * 100, b * 100

    # Convert to XYZ (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize for D65 white point
    x, y, z = x / 95.047, y / 100.0, z / 108.883

    # Apply Lab conversion
    epsilon = 0.008856
    kappa = 903.3

    fx = x ** (1/3) if x > epsilon else (kappa * x + 16) / 116
    fy = y ** (1/3) if y > epsilon else (kappa * y + 16) / 116
    fz = z ** (1/3) if z > epsilon else (kappa * z + 16) / 116

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_lab = 200 * (fy - fz)

    return L, a, b_lab

def lab_distance(lab1, lab2):
    """Calculate Euclidean distance in Lab space."""
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return math.sqrt(dL * dL + da * da + db * db)

# Pre-compute Lab values for the palette
PALETTE_LAB = [rgb_to_lab(r, g, b) for r, g, b in PALETTE_RGB]

def find_closest_color(r, g, b):
    """Find the closest palette color using Lab distance."""
    pixel_lab = rgb_to_lab(r, g, b)
    min_dist = float('inf')
    closest_idx = 0

    for idx, palette_lab in enumerate(PALETTE_LAB):
        dist = lab_distance(pixel_lab, palette_lab)
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx

    return closest_idx

def floyd_steinberg_dither_lab(img_array, strength=1.0, quiet=False):
    """
    Apply Floyd-Steinberg dithering using Lab color space for color matching.

    Args:
        img_array: numpy array of shape (height, width, 3) with RGB values (0-255)
        strength: dithering strength (1.0 = full, 0.0 = none)
        quiet: suppress progress messages

    Returns:
        numpy array of palette indices
    """
    height, width = img_array.shape[:2]
    # Work with float array to handle error diffusion
    pixels = img_array.astype(np.float32)
    result = np.zeros((height, width), dtype=np.uint8)

    if not quiet:
        print("  Dithering...")

    for y in range(height):
        if not quiet and y % 400 == 0:
            print(f"    Row {y}/{height}...")
        for x in range(width):
            # Get current pixel
            r, g, b = pixels[y, x]
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            # Find closest palette color using Lab distance
            closest_idx = find_closest_color(r, g, b)
            result[y, x] = closest_idx

            # Get the palette color RGB
            new_r, new_g, new_b = PALETTE_RGB[closest_idx]

            # Calculate quantization error
            err_r = (r - new_r) * strength
            err_g = (g - new_g) * strength
            err_b = (b - new_b) * strength

            # Distribute error to neighboring pixels (Floyd-Steinberg pattern)
            # Right: 7/16
            if x + 1 < width:
                pixels[y, x + 1, 0] += err_r * 7 / 16
                pixels[y, x + 1, 1] += err_g * 7 / 16
                pixels[y, x + 1, 2] += err_b * 7 / 16

            # Bottom-left: 3/16
            if y + 1 < height and x > 0:
                pixels[y + 1, x - 1, 0] += err_r * 3 / 16
                pixels[y + 1, x - 1, 1] += err_g * 3 / 16
                pixels[y + 1, x - 1, 2] += err_b * 3 / 16

            # Bottom: 5/16
            if y + 1 < height:
                pixels[y + 1, x, 0] += err_r * 5 / 16
                pixels[y + 1, x, 1] += err_g * 5 / 16
                pixels[y + 1, x, 2] += err_b * 5 / 16

            # Bottom-right: 1/16
            if y + 1 < height and x + 1 < width:
                pixels[y + 1, x + 1, 0] += err_r * 1 / 16
                pixels[y + 1, x + 1, 1] += err_g * 1 / 16
                pixels[y + 1, x + 1, 2] += err_b * 1 / 16

    return result

# --- IMAGE PROCESSING ---

def load_and_resize_image(image_path):
    """Load image and resize to frame dimensions."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    return ImageOps.fit(img, (FRAME_WIDTH, FRAME_HEIGHT), method=Image.Resampling.LANCZOS)

def apply_enhancements(img, contrast=1.0, brightness=1.0, saturation=1.0):
    """Apply contrast, brightness, and saturation enhancements."""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    return img

def process_with_enhancements(img, contrast, brightness, saturation, quiet=False):
    """Process image with specific enhancement settings, return palette indices."""
    enhanced = apply_enhancements(img.copy(), contrast, brightness, saturation)
    img_array = np.array(enhanced)
    return floyd_steinberg_dither_lab(img_array, quiet=quiet)

# --- COMPARISON FUNCTIONS ---

def compare_results(indices_a, indices_b):
    """Compare two dithered results and return statistics."""
    # Find where they differ
    diff_mask = indices_a != indices_b
    changed_count = np.sum(diff_mask)
    total_pixels = indices_a.size

    # Count transitions
    transitions = defaultdict(int)
    if changed_count > 0:
        changed_a = indices_a[diff_mask]
        changed_b = indices_b[diff_mask]
        for a, b in zip(changed_a, changed_b):
            key = (int(a), int(b))
            transitions[key] += 1

    # Count color distribution for each config
    color_counts_a = {i: int(np.sum(indices_a == i)) for i in range(6)}
    color_counts_b = {i: int(np.sum(indices_b == i)) for i in range(6)}

    return {
        'total_pixels': total_pixels,
        'changed_count': int(changed_count),
        'changed_percent': 100.0 * changed_count / total_pixels,
        'transitions': dict(transitions),
        'diff_mask': diff_mask,
        'color_counts_a': color_counts_a,
        'color_counts_b': color_counts_b
    }

def print_comparison_report(stats, config_a, config_b):
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("ENHANCEMENT COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nConfig A: contrast={config_a[0]}, brightness={config_a[1]}, saturation={config_a[2]}")
    print(f"Config B: contrast={config_b[0]}, brightness={config_b[1]}, saturation={config_b[2]}")
    print(f"\nTotal pixels: {stats['total_pixels']:,}")
    print(f"Pixels changed: {stats['changed_count']:,} ({stats['changed_percent']:.2f}%)")

    # Color distribution comparison
    print("\nColor Distribution:")
    print(f"  {'Color':<8} {'Config A':>12} {'Config B':>12} {'Change':>12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for idx in range(6):
        count_a = stats['color_counts_a'][idx]
        count_b = stats['color_counts_b'][idx]
        change = count_b - count_a
        sign = "+" if change > 0 else ""
        print(f"  {COLOR_NAMES[idx]:<8} {count_a:>12,} {count_b:>12,} {sign}{change:>11,}")

    if stats['transitions']:
        print("\nTop Transitions (A -> B):")
        # Sort by count descending, show top 15
        sorted_trans = sorted(stats['transitions'].items(), key=lambda x: -x[1])[:15]
        for (from_idx, to_idx), count in sorted_trans:
            from_name = COLOR_NAMES[from_idx]
            to_name = COLOR_NAMES[to_idx]
            print(f"  {from_name:8} -> {to_name:8}: {count:,}")
    else:
        print("\nNo color transitions detected - results are identical!")

    print("=" * 70 + "\n")

def indices_to_image(indices):
    """Convert palette indices to RGB image."""
    height, width = indices.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, rgb in enumerate(PALETTE_RGB):
        rgb_array[indices == idx] = rgb
    return Image.fromarray(rgb_array)

def create_diff_image(diff_mask):
    """Create an image highlighting where differences occurred."""
    height, width = diff_mask.shape
    # Magenta for changed pixels, dark gray for unchanged
    diff_array = np.zeros((height, width, 3), dtype=np.uint8)
    diff_array[~diff_mask] = [40, 40, 40]  # Dark gray for unchanged
    diff_array[diff_mask] = [255, 0, 255]  # Magenta for changed
    return Image.fromarray(diff_array)

def save_images(indices_a, indices_b, stats, save_diff=False):
    """Save comparison images to /tmp/."""
    print("Saving images...")

    img_a = indices_to_image(indices_a)
    img_a.save("/tmp/enhance_configA.png")
    print("  Saved /tmp/enhance_configA.png")

    img_b = indices_to_image(indices_b)
    img_b.save("/tmp/enhance_configB.png")
    print("  Saved /tmp/enhance_configB.png")

    if save_diff:
        diff_img = create_diff_image(stats['diff_mask'])
        diff_img.save("/tmp/enhance_diff.png")
        print("  Saved /tmp/enhance_diff.png")

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare how enhancement parameters affect dithering results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg
  %(prog)s photo.jpg --contrast1 1.5 --contrast2 0.8 --save-images
  %(prog)s photo.jpg --saturation1 1.5 --saturation2 0.5 --save-images --diff-image
        """
    )

    parser.add_argument("image", help="Path to the image file")

    # Config A enhancements (defaults match send_to_display.py defaults)
    parser.add_argument("--contrast1", type=float, default=1.2,
                        help="Contrast for config A (default: 1.2)")
    parser.add_argument("--brightness1", type=float, default=1.0,
                        help="Brightness for config A (default: 1.0)")
    parser.add_argument("--saturation1", type=float, default=1.2,
                        help="Saturation for config A (default: 1.2)")

    # Config B enhancements (defaults to no enhancement)
    parser.add_argument("--contrast2", type=float, default=1.0,
                        help="Contrast for config B (default: 1.0)")
    parser.add_argument("--brightness2", type=float, default=1.0,
                        help="Brightness for config B (default: 1.0)")
    parser.add_argument("--saturation2", type=float, default=1.0,
                        help="Saturation for config B (default: 1.0)")

    # Output options
    parser.add_argument("--save-images", action="store_true",
                        help="Save preview images to /tmp/")
    parser.add_argument("--diff-image", action="store_true",
                        help="Create difference image highlighting changed pixels")

    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    base_img = load_and_resize_image(args.image)

    config_a = (args.contrast1, args.brightness1, args.saturation1)
    config_b = (args.contrast2, args.brightness2, args.saturation2)

    print(f"\nProcessing with Config A (c={args.contrast1}, b={args.brightness1}, s={args.saturation1})...")
    indices_a = process_with_enhancements(base_img, *config_a)

    print(f"\nProcessing with Config B (c={args.contrast2}, b={args.brightness2}, s={args.saturation2})...")
    indices_b = process_with_enhancements(base_img, *config_b)

    print("\nComparing results...")
    stats = compare_results(indices_a, indices_b)

    print_comparison_report(stats, config_a, config_b)

    if args.save_images:
        save_images(indices_a, indices_b, stats, save_diff=args.diff_image)
