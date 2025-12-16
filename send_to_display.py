import sys
import time
import argparse
import requests
import subprocess
import os
from PIL import Image, ImageOps, ImageEnhance

# --- DOCUMENTATION ---
# Dependencies: uv add requests pillow
# Usage (Manual): uv run send_to_frame.py my_photo.jpg --show --contrast 1.5
# Usage (Cron):   uv run send_to_frame.py my_photo.jpg

# --- CONFIGURATION ---
ESP32_IP = "192.168.86.24"
FRAME_WIDTH = 1200
FRAME_HEIGHT = 1600
CHUNK_SIZE = 960000

# The Spectra 6 Color Palette (RGB)
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

def process_image(image_path, contrast=1.0, brightness=1.0, saturation=1.0, show_preview=False):
    print(f"Processing {image_path}...")
    print(f"Settings -> Contrast: {contrast}, Brightness: {brightness}, Saturation: {saturation}")
    
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

    # 3. Create Palette for Dithering
    flat_palette = [c for color in PALETTE_RGB for c in color]
    flat_palette += [0] * (768 - len(flat_palette)) # Pad to 768 bytes
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)

    # 4. Dither using Floyd-Steinberg
    dithered = img.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    
    # --- CONDITIONAL PREVIEW ---
    if show_preview:
        preview_path = "/tmp/spectra_preview.png"
        dithered.save(preview_path)
        print(f"Preview saved to {preview_path}")
        try:
            print("Launching 'display' (ImageMagick)...")
            # We use subprocess to force the 'display' command, bypassing PIL's viewer logic
            subprocess.run(["display", preview_path])
        except FileNotFoundError:
            print("Error: 'display' command not found. Install ImageMagick.")
        except Exception as e:
            print(f"Error launching preview: {e}")
    # ---------------------------

    # 5. Pack bits (2 pixels per byte)
    pixels = list(dithered.getdata())
    packed_data = bytearray()
    
    print("Packing binary data...")
    for i in range(0, len(pixels), 2):
        p1_idx = pixels[i]
        # Handle case where pixel count is odd (though 1200x1600 is even)
        p2_idx = pixels[i+1] if i+1 < len(pixels) else 0 
        
        val1 = HARDWARE_MAP.get(p1_idx, 0x01) # Default to White
        val2 = HARDWARE_MAP.get(p2_idx, 0x01)
        
        # Combine: High Nibble | Low Nibble
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
    
    # Optional Arguments
    parser.add_argument("-c", "--contrast", type=float, default=1.2, help="Contrast (1.0 = original)")
    parser.add_argument("-b", "--brightness", type=float, default=1.0, help="Brightness (1.0 = original)")
    parser.add_argument("-s", "--saturation", type=float, default=1.2, help="Saturation (1.0 = original, 0.0 = B&W)")
    
    # The flag that pauses the script to show the image
    parser.add_argument("--show", action="store_true", help="Pop up a preview window using ImageMagick 'display'")

    args = parser.parse_args()
    
    try:
        data = process_image(
            args.image, 
            contrast=args.contrast, 
            brightness=args.brightness, 
            saturation=args.saturation,
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
