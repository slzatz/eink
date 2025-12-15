import sys
import time
import argparse
import requests
import subprocess  # <--- Add this
import os          # <--- Add this (for temp file cleanup)
from PIL import Image, ImageOps, ImageEnhance

# --- CONFIGURATION ---
ESP32_IP = "192.168.86.34"
FRAME_WIDTH = 1200
FRAME_HEIGHT = 1600
CHUNK_SIZE = 960000

# The Spectra 6 Color Palette
PALETTE_RGB = [
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (255, 255, 0),   # Yellow
    (255, 0, 0),     # Red
    (0, 0, 255),     # Blue
    (41, 204, 20)    # Green
]

HARDWARE_MAP = {
    0: 0x00, 1: 0x01, 2: 0x02, 3: 0x03, 4: 0x05, 5: 0x06
}

def process_image(image_path, contrast=1.0, brightness=1.0, saturation=1.0):
    print(f"Processing {image_path}...")
    print(f"Settings -> Contrast: {contrast}, Brightness: {brightness}, Saturation: {saturation}")
    
    # 1. Open and Resize
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (FRAME_WIDTH, FRAME_HEIGHT), method=Image.Resampling.LANCZOS)

    # 2. Apply Adjustments (The "Sliders")
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
    flat_palette += [0] * (768 - len(flat_palette))
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)

    # 4. Dither (Floyd-Steinberg is standard in PIL)
    dithered = img.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    #dithered.show()
    # --- FORCE DISPLAY PREVIEW ---
    # We know 'display' works, so we call it manually
    preview_path = "/tmp/spectra_preview.png"
    dithered.save(preview_path)
    try:
        print("Launching ImageMagick 'display'...")
        # This runs the 'display' command exactly like you did in the terminal
        subprocess.run(["display", preview_path]) 
    except FileNotFoundError:
        print("Error: Could not find 'display' command.")
    # -----------------------------    
    
    # 5. Pack bits
    pixels = list(dithered.getdata())
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
    total_chunks = (total_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Uploading {total_len} bytes...")

    with requests.Session() as session:
        for i in range(total_chunks):
            start = i * CHUNK_SIZE
            end = min((i + 1) * CHUNK_SIZE, total_len)
            chunk = binary_data[start:end]
            files = {'data': ('image_data.bin', chunk, 'application/octet-stream')}
            
            try:
                response = session.post(url, files=files)
                response.raise_for_status()
                print(f"Chunk {i+1}/{total_chunks} sent.")
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
                return False
            time.sleep(1.0) 

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload resized/dithered image to Spectra 6 Frame")
    parser.add_argument("image", help="Path to the image file")
    
    # Optional arguments (The Sliders!)
    parser.add_argument("-c", "--contrast", type=float, default=1.2, help="Contrast (1.0 = original, 1.5 = high, 0.8 = low)")
    parser.add_argument("-b", "--brightness", type=float, default=1.0, help="Brightness (1.0 = original, 1.2 = brighter)")
    parser.add_argument("-s", "--saturation", type=float, default=1.2, help="Color Saturation (1.0 = original, 0.0 = B&W)")

    args = parser.parse_args()
    
    try:
        data = process_image(args.image, contrast=args.contrast, brightness=args.brightness, saturation=args.saturation)
        success = upload_to_esp32(data)
        if success:
            print("Done! Check your frame.")
    except Exception as e:
        print(f"Failed: {e}")
