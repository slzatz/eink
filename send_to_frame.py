import sys
import time
import requests
from PIL import Image, ImageOps

# --- CONFIGURATION ---
ESP32_IP = "192.168.86.34"  # Your ESP32 IP
FRAME_WIDTH = 1200
FRAME_HEIGHT = 1600
CHUNK_SIZE = 960000         # Matches the JS chunk size

# The Spectra 6 Color Palette (RGB)
# These match the colors defined in your JavaScript
PALETTE_RGB = [
    (0, 0, 0),       # Black
    (255, 255, 255), # White
    (255, 255, 0),   # Yellow
    (255, 0, 0),     # Red
    (0, 0, 255),     # Blue
    (41, 204, 20)    # Green
]

# The Hardware Values for each color
# The screen expects specific 4-bit codes for each color, not just 0-5.
# Based on the JS 'rgbToSixColor' function:
# Black=0x00, White=0x01, Yellow=0x02, Red=0x03, Blue=0x05, Green=0x06
HARDWARE_MAP = {
    0: 0x00, # Index 0 (Black)  -> 0x00
    1: 0x01, # Index 1 (White)  -> 0x01
    2: 0x02, # Index 2 (Yellow) -> 0x02
    3: 0x03, # Index 3 (Red)    -> 0x03
    4: 0x05, # Index 4 (Blue)   -> 0x05
    5: 0x06  # Index 5 (Green)  -> 0x06
}

def process_image(image_path):
    print(f"Processing {image_path}...")
    
    # 1. Open and Resize/Crop to fit 1200x1600
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (FRAME_WIDTH, FRAME_HEIGHT), method=Image.Resampling.LANCZOS)

    # 2. Create a Palette Image for Dithering
    # We flatten the list of tuples into a single list: [0,0,0, 255,255,255, ...]
    flat_palette = [c for color in PALETTE_RGB for c in color]
    # Pad the palette to 768 bytes (256 colors * 3) required by PIL
    flat_palette += [0] * (768 - len(flat_palette))
    
    palette_img = Image.new('P', (1, 1))
    palette_img.putpalette(flat_palette)

    # 3. Dither the image
    # This replaces Floyd-Steinberg in JS. PIL's dithering is very fast/good.
    dithered = img.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
    
    # 4. Pack bits (Six Color Mode)
    # The JS packs 2 pixels into 1 byte.
    # Pixel 1 occupies the high 4 bits, Pixel 2 the low 4 bits.
    pixels = list(dithered.getdata())
    packed_data = bytearray()
    
    print("Packing binary data...")
    
    # Iterate 2 pixels at a time
    for i in range(0, len(pixels), 2):
        p1_idx = pixels[i]
        p2_idx = pixels[i+1] if i+1 < len(pixels) else 0 # Handle odd pixel count if necessary
        
        # Map palette index to hardware nibble
        val1 = HARDWARE_MAP.get(p1_idx, 0x01) # Default to white if error
        val2 = HARDWARE_MAP.get(p2_idx, 0x01)
        
        # Combine: (val1 << 4) | val2
        byte_val = (val1 << 4) | val2
        packed_data.append(byte_val)
        
    return packed_data

def upload_to_esp32(binary_data):
    url = f"http://{ESP32_IP}/upload"
    total_len = len(binary_data)
    total_chunks = (total_len + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    print(f"Uploading {total_len} bytes in {total_chunks} chunks to {url}...")

    # We use a session to keep the connection alive
    with requests.Session() as session:
        for i in range(total_chunks):
            start = i * CHUNK_SIZE
            end = min((i + 1) * CHUNK_SIZE, total_len)
            chunk = binary_data[start:end]
            
            # Prepare multipart upload
            files = {
                'data': ('image_data.bin', chunk, 'application/octet-stream')
            }
            
            try:
                # Note: No headers needed, requests handles multipart boundary automatically
                response = session.post(url, files=files)
                response.raise_for_status() # Raise error if not 200 OK
                print(f"Chunk {i+1}/{total_chunks} sent successfully.")
            except requests.exceptions.RequestException as e:
                print(f"Error uploading chunk {i+1}: {e}")
                return False
            
            # Small delay to let ESP32 write to flash (matches JS logic)
            time.sleep(1.0) 

    print("Upload Complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python send_to_frame.py <image_file>")
        sys.exit(1)
        
    image_file = sys.argv[1]
    
    try:
        data = process_image(image_file)
        success = upload_to_esp32(data)
        if success:
            print("Success! Check your frame.")
        else:
            print("Failed to upload.")
    except Exception as e:
        print(f"An error occurred: {e}")
