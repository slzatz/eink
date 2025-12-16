import sys
import time
import argparse
import requests
import subprocess
import math
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from config import ESP32_IP

# --- DOCUMENTATION ---
# Dependencies: uv add requests pillow numpy
# Usage (Manual): uv run send_to_display.py my_photo.jpg --show --contrast 1.5
# Usage (Cron):   uv run send_to_display.py my_photo.jpg

# --- CONFIGURATION ---
# ESP32_IP is imported from config.py
FRAME_WIDTH = 1200
FRAME_HEIGHT = 1600
CHUNK_SIZE = 960000

# The Spectra 6 Color Palette (RGB)
# NOTE: These are idealized values. For best results, measure the actual colors
# your display produces and update these values accordingly.
PALETTE_RGB = [
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (255, 255, 0),   # Yellow
    (255, 0, 0),     # Red
    (0, 0, 255),     # Blue
    (41, 204, 20)    # Green
]

# Map palette index to hardware 4-bit codes
HARDWARE_MAP = {
    0: 0x00, # Black
    1: 0x01, # White
    2: 0x02, # Yellow
    3: 0x03, # Red
    4: 0x05, # Blue
    5: 0x06  # Green
}

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

def lab_distance(lab1, lab2, luminance_weight, chroma_weight):
    """
    Calculate weighted distance in Lab space.
    Lower luminance_weight emphasizes color over brightness.
    Higher chroma_weight makes color differences more important.
    """
    dL = lab1[0] - lab2[0]
    da = lab1[1] - lab2[1]
    db = lab1[2] - lab2[2]
    return math.sqrt(luminance_weight * dL * dL + chroma_weight * da * da + chroma_weight * db * db)

# Pre-compute Lab values for the palette
PALETTE_LAB = [rgb_to_lab(r, g, b) for r, g, b in PALETTE_RGB]

def find_closest_color(r, g, b, luminance_weight, chroma_weight):
    """Find the closest palette color using Lab distance."""
    pixel_lab = rgb_to_lab(r, g, b)
    min_dist = float('inf')
    closest_idx = 0

    for idx, palette_lab in enumerate(PALETTE_LAB):
        dist = lab_distance(pixel_lab, palette_lab, luminance_weight, chroma_weight)
        if dist < min_dist:
            min_dist = dist
            closest_idx = idx

    return closest_idx

def floyd_steinberg_dither_lab(img_array, luminance_weight=0.2, chroma_weight=3.0, strength=1.0):
    """
    Apply Floyd-Steinberg dithering using Lab color space for color matching.

    Args:
        img_array: numpy array of shape (height, width, 3) with RGB values (0-255)
        luminance_weight: weight for luminance differences (default 0.2)
        chroma_weight: weight for color differences (default 3.0)
        strength: dithering strength (1.0 = full, 0.0 = none)

    Returns:
        numpy array of palette indices
    """
    height, width = img_array.shape[:2]
    # Work with float array to handle error diffusion
    pixels = img_array.astype(np.float32)
    result = np.zeros((height, width), dtype=np.uint8)

    print(f"Applying Lab-based Floyd-Steinberg dithering (lum={luminance_weight}, chroma={chroma_weight})...")

    for y in range(height):
        if y % 200 == 0:
            print(f"  Processing row {y}/{height}...")
        for x in range(width):
            # Get current pixel
            r, g, b = pixels[y, x]
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            # Find closest palette color using Lab distance
            closest_idx = find_closest_color(r, g, b, luminance_weight, chroma_weight)
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

def process_image(image_path, contrast=1.0, brightness=1.0, saturation=1.0,
                  luminance_weight=0.2, chroma_weight=3.0, show_preview=False):
    print(f"Processing {image_path}...")
    print(f"Settings -> Contrast: {contrast}, Brightness: {brightness}, Saturation: {saturation}")
    print(f"Lab weights -> Luminance: {luminance_weight}, Chroma: {chroma_weight}")

    # 1. Open and Resize/Crop to fit 1200x1600
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    img = ImageOps.fit(img, (FRAME_WIDTH, FRAME_HEIGHT), method=Image.Resampling.LANCZOS)

    # 2. Apply Image Enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    # 3. Convert to numpy array and apply Lab-based Floyd-Steinberg dithering
    img_array = np.array(img)
    dithered_indices = floyd_steinberg_dither_lab(img_array, luminance_weight, chroma_weight)

    # --- CONDITIONAL PREVIEW ---
    if show_preview:
        # Create preview image from palette indices
        preview_array = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for idx, rgb in enumerate(PALETTE_RGB):
            preview_array[dithered_indices == idx] = rgb
        preview_img = Image.fromarray(preview_array)

        preview_path = "/tmp/spectra_preview.png"
        preview_img.save(preview_path)
        print(f"Preview saved to {preview_path}")
        try:
            print("Launching 'display' (ImageMagick)...")
            subprocess.run(["display", preview_path])
        except FileNotFoundError:
            print("Error: 'display' command not found. Install ImageMagick.")
        except Exception as e:
            print(f"Error launching preview: {e}")
    # ---------------------------

    # 4. Pack bits (2 pixels per byte)
    pixels = dithered_indices.flatten()
    packed_data = bytearray()

    print("Packing binary data...")
    for i in range(0, len(pixels), 2):
        p1_idx = pixels[i]
        p2_idx = pixels[i+1] if i+1 < len(pixels) else 0

        val1 = HARDWARE_MAP.get(p1_idx, 0x01)
        val2 = HARDWARE_MAP.get(p2_idx, 0x01)

        byte_val = (val1 << 4) | val2
        packed_data.append(byte_val)

    return packed_data

def upload_to_esp32(binary_data):
    url = f"http://{ESP32_IP}/upload"
    total_len = len(binary_data)
    print(f"Total binary data size: {total_len} bytes")
    total_chunks = (total_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Uploading {total_len} bytes to {url}...")

    with requests.Session() as session:
        for i in range(total_chunks):
            start = i * CHUNK_SIZE
            end = min((i + 1) * CHUNK_SIZE, total_len)
            chunk = binary_data[start:end]
            
            # ESP32 expects 'image_data.bin' as the filename in the form data
            files = {'data': ('image_data.bin', chunk, 'application/octet-stream')}
            
            try:
                response = session.post(url, files=files)
                response.raise_for_status()
                print(f"Chunk {i+1}/{total_chunks} sent.")
            except requests.exceptions.RequestException as e:
                print(f"Network Error during upload: {e}")
                return False
            
            # Small delay to ensure ESP32 writes to flash without buffer overflow
            time.sleep(1.0) 

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload resized/dithered image to Spectra 6 Frame")
    parser.add_argument("image", help="Path to the image file")
    
    # Optional Arguments - Image Enhancement
    parser.add_argument("-c", "--contrast", type=float, default=1.2, help="Contrast (1.0 = original)")
    parser.add_argument("-b", "--brightness", type=float, default=1.0, help="Brightness (1.0 = original)")
    parser.add_argument("-s", "--saturation", type=float, default=1.2, help="Saturation (1.0 = original, 0.0 = B&W)")

    # Optional Arguments - Lab Color Matching Weights
    parser.add_argument("-l", "--luminance", type=float, default=0.2,
                        help="Luminance weight for color matching (default: 0.2, range: 0.0-2.0)")
    parser.add_argument("-C", "--chroma", type=float, default=3.0,
                        help="Chroma weight for color matching (default: 3.0, range: 0.5-5.0)")

    # The flag that pauses the script to show the image
    parser.add_argument("--show", action="store_true", help="Pop up a preview window using ImageMagick 'display'")

    args = parser.parse_args()
    
    try:
        data = process_image(
            args.image,
            contrast=args.contrast,
            brightness=args.brightness,
            saturation=args.saturation,
            luminance_weight=args.luminance,
            chroma_weight=args.chroma,
            show_preview=args.show
        )
        
        print("Starting Upload...")
        success = upload_to_esp32(data)
        
        if success:
            print("--------------------------------")
            print("Success! Image sent to frame.")
            print("--------------------------------")
        else:
            print("Upload failed.")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
