import os
import shutil
import subprocess
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageChops

# --- CONFIGURATION ---
SETTINGS = {
    'fps': 24,
    'transition_duration': 0.25, # Percentage of total time
    'pixel_sort_threshold': 0.4, # How likely to sort pixels
    'mosh_threshold': 15,        # Threshold (0-255) for pixel updates (Higher = More Smear)
    'bloom_intensity': 0.3       # Chance to apply heavy compression artifacts
}

# --- FFMPEG WRAPPERS ---

def run_ffmpeg(command):
    """Runs ffmpeg with robust error handling."""
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as e:
        # We don't raise immediately; we let the caller handle specific fallbacks
        raise e

def safe_extract_frames(input_path, output_folder, target_fps=24):
    """
    Attempts to extract frames using FFmpeg. 
    If that fails (e.g., input is an image renamed to mp4), 
    it falls back to PIL to generate a static sequence.
    """
    print(f"Processing input: {input_path}")
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Try standard Video Extraction
    try:
        run_ffmpeg([
            "ffmpeg", "-i", input_path, "-vf", f"fps={target_fps}", 
            os.path.join(output_folder, "frame_%05d.png")
        ])
        print(" -> Identified as Video. Frames extracted.")
        return
    except subprocess.CalledProcessError:
        print(" -> FFmpeg failed. Attempting to treat as Static Image...")

    # 2. Fallback: Treat as Static Image
    try:
        img = Image.open(input_path).convert("RGB")
        # Generate 3 seconds worth of frames for a static image
        count = target_fps * 3 
        print(f" -> Identified as Image. Generating {count} static frames...")
        for i in range(count):
            img.save(os.path.join(output_folder, f"frame_{i:05d}.png"))
    except Exception as e:
        print(f"FATAL: Could not read file as video OR image. {e}")
        raise e

def extract_and_glitch_audio(video_path, output_audio):
    """Extracts audio, adds reverb/distortion."""
    temp_raw = "temp_raw_audio.aac"
    if os.path.exists(temp_raw): os.remove(temp_raw)
    
    # Try extraction
    try:
        run_ffmpeg(["ffmpeg", "-i", video_path, "-vn", "-c:a", "aac", temp_raw])
    except:
        return False # No audio found

    # Apply Glitch Filters (Vibrato + Echo + Tempo Slow)
    try:
        run_ffmpeg([
            "ffmpeg", "-y", "-i", temp_raw, "-af", 
            "aecho=0.8:0.9:1000:0.3,vibrato=f=6.0:d=0.5,atempo=0.9",
            "-c:a", "aac", output_audio
        ])
        return True
    except:
        return False

# --- DATAMOSH EFFECT SIMULATIONS ---

def pixel_sort(image, probability=0.5):
    """Randomly sorts rows of pixels based on luminance."""
    if random.random() > probability: return image

    img_arr = np.array(image)
    
    # Decide: Horizontal or Vertical sort?
    if random.random() > 0.5:
        img_arr = np.swapaxes(img_arr, 0, 1) # Rotate for vertical processing
        
    rows, cols, _ = img_arr.shape
    
    # Sort a random chunk of rows
    start_row = random.randint(0, rows - 5)
    end_row = random.randint(start_row + 4, rows)
    
    # Vectorized sorting is hard, doing row-by-row for simplicity/effect
    for i in range(start_row, end_row):
        row = img_arr[i]
        # Calculate luminance for sorting key
        lum = np.dot(row[...,:3], [0.299, 0.587, 0.114])
        # Sort indices
        sorted_indices = np.argsort(lum)
        img_arr[i] = row[sorted_indices]

    if random.random() > 0.5:
        img_arr = np.swapaxes(img_arr, 0, 1) # Rotate back
        
    return Image.fromarray(img_arr)

def simulated_p_frame_mosh(current_frame, previous_frame, threshold=15):
    """
    The 'Datamosh' look:
    Compare current frame to previous. 
    If a pixel hasn't changed *enough* (below threshold), 
    keep the PREVIOUS pixel (ghosting/smearing).
    """
    if previous_frame is None: return current_frame
    
    curr_arr = np.array(current_frame).astype(int)
    prev_arr = np.array(previous_frame).astype(int)
    
    # Calculate absolute difference per channel
    diff = np.abs(curr_arr - prev_arr)
    # Sum differences across RGB
    total_diff = np.sum(diff, axis=2)
    
    # Create mask: Where is difference < threshold?
    mask = total_diff < threshold
    
    # Apply mask: Where mask is True (low motion), use previous frame's pixels
    # This creates the "stuck" pixel effect
    curr_arr[mask] = prev_arr[mask]
    
    return Image.fromarray(curr_arr.astype('uint8'))

def jpeg_bloom(image, intensity=0.5):
    """Saves and re-opens image at low quality to create block artifacts."""
    if random.random() > intensity: return image
    
    import io
    buffer = io.BytesIO()
    # Save with low quality
    image.save(buffer, format="JPEG", quality=random.randint(5, 20))
    buffer.seek(0)
    return Image.open(buffer)

# --- MAIN WORKFLOW ---

def main():
    v1_path = "input.mp4"
    v2_path = "image2.mp4"
    final_output = "output_horrifying_mosh.mp4"
    
    # Temp folders
    t_v1 = "frames_v1"
    t_v2 = "frames_v2"
    t_out = "frames_out"
    t_audio = "glitched_audio.aac"
    
    # Cleanup start
    for d in [t_v1, t_v2, t_out]:
        if os.path.exists(d): shutil.rmtree(d)
        
    try:
        # 1. EXTRACTION (With Fallback)
        safe_extract_frames(v1_path, t_v1, SETTINGS['fps'])
        safe_extract_frames(v2_path, t_v2, SETTINGS['fps'])
        
        # Audio (Try v1 first)
        has_audio = extract_and_glitch_audio(v1_path, t_audio)
        
        # 2. PROCESSING LOOP
        files_v1 = sorted([os.path.join(t_v1, f) for f in os.listdir(t_v1) if f.endswith('.png')])
        files_v2 = sorted([os.path.join(t_v2, f) for f in os.listdir(t_v2) if f.endswith('.png')])
        
        if not files_v1 and not files_v2:
            raise Exception("No frames extracted from either input.")
            
        # Combine lists based on transition
        # We append v2 to v1, but we blend them during the transition
        total_frames = len(files_v1) + len(files_v2)
        os.makedirs(t_out, exist_ok=True)
        
        print(f"Generating {total_frames} frames of destruction...")
        
        previous_img = None
        
        for i in range(total_frames):
            print(f"Rendering frame {i}/{total_frames}", end='\r')
            
            # Determine Source Image
            if i < len(files_v1):
                src_img = Image.open(files_v1[i]).convert("RGB")
                
                # If we are near the end of v1, blend with start of v2
                frames_left = len(files_v1) - i
                transition_len = int(len(files_v1) * SETTINGS['transition_duration'])
                
                if frames_left < transition_len and files_v2:
                    # Map progress to v2
                    v2_idx = int((1 - (frames_left / transition_len)) * (len(files_v2)-1))
                    img_v2 = Image.open(files_v2[v2_idx]).convert("RGB")
                    # Chaotic blend
                    src_img = ImageChops.difference(src_img, img_v2)
            else:
                # We are in V2 territory
                v2_idx = i - len(files_v1)
                if v2_idx >= len(files_v2): break
                src_img = Image.open(files_v2[v2_idx]).convert("RGB")
            
            # --- APPLY EFFECTS ---
            
            # 1. P-Frame Simulation (The Mosh)
            # We don't apply this on the VERY first frame
            if previous_img is not None:
                src_img = simulated_p_frame_mosh(src_img, previous_img, SETTINGS['mosh_threshold'])
            
            # 2. Pixel Sort (The Glitch)
            src_img = pixel_sort(src_img, SETTINGS['pixel_sort_threshold'])
            
            # 3. JPEG Bloom (The Decay)
            src_img = jpeg_bloom(src_img, SETTINGS['bloom_intensity'])

            # Save
            save_path = os.path.join(t_out, f"frame_{i:05d}.png")
            src_img.save(save_path)
            
            # Update history for next P-frame comparison
            previous_img = src_img

        print("\nRendering complete. Assembling video...")

        # 3. ASSEMBLY
        cmd = [
            "ffmpeg", "-y", "-framerate", str(SETTINGS['fps']),
            "-i", os.path.join(t_out, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "temp_video.mp4"
        ]
        run_ffmpeg(cmd)
        
        # Merge Audio if exists
        if has_audio and os.path.exists(t_audio):
            run_ffmpeg([
                "ffmpeg", "-y", "-i", "temp_video.mp4", "-i", t_audio,
                "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", final_output
            ])
        else:
            os.rename("temp_video.mp4", final_output)
            
        print(f"\nDONE. Output saved to {final_output}")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        # Create a dummy file so GitHub Actions upload doesn't fail entirely (helps debugging)
        with open("error_log.txt", "w") as f: f.write(str(e))
        exit(1)

if __name__ == "__main__":
    main()
