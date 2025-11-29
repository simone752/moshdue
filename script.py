#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import random
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps

# --- CONFIGURATION ---
SETTINGS = {
    "fps": 24,
    "internal_res": (640, 360), # Process low-res for SPEED + BLOCKY AESTHETIC
    "transition_frac": 0.2,     # How fast videos blend
    
    # Chaos Parameters
    "mosh_threshold": 12,       # Lower = More motion required to update pixel (More Smear)
    "bloom_chance": 0.08,       # Chance to explode compression artifacts
    "sort_chance": 0.25,        # Chance to melt the screen
    "invert_chance": 0.02,      # Chance to flash negative
    "stutter_chance": 0.23,     # Chance to freeze frame
}

# --- FFMPEG HELPERS ---
def run_ffmpeg(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr}")
        raise e

def get_audio_filter():
    """Generates a harsh, abrasive audio filter chain."""
    return "acrusher=level_in=8:level_out=18:bits=8:mode=log:aa=1, vibrato=f=7:d=0.5, aecho=0.8:0.8:20:0.6"

# --- VECTORIZED VISUAL EFFECTS ---

def vector_pixel_sort(img_arr):
    """
    Fast, row-based pixel sorting using numpy.
    Sorts pixels by brightness, but only in high-contrast areas.
    """
    # Create a sort key (luminance)
    lum = np.dot(img_arr[...,:3], [0.299, 0.587, 0.114])
    
    # Decide direction (Horizontal vs Vertical)
    if random.random() > 0.5:
        # Sort based on indices
        sorted_indices = np.argsort(lum, axis=1)
        # Apply sort to image
        return np.take_along_axis(img_arr, np.expand_dims(sorted_indices, axis=2), axis=1)
    else:
        sorted_indices = np.argsort(lum, axis=0)
        return np.take_along_axis(img_arr, np.expand_dims(sorted_indices, axis=2), axis=0)

def vector_mosh(current_arr, previous_arr, threshold):
    """
    The True 'YouTube' Effect.
    Compare Current INPUT vs Previous OUTPUT.
    If difference < threshold, KEEP PREVIOUS OUTPUT (Smear).
    """
    if previous_arr is None: return current_arr
    
    # Calculate Euclidean distance between pixels (vectorized)
    diff = np.sum(np.abs(current_arr.astype(int) - previous_arr.astype(int)), axis=2)
    
    # Create Boolean Mask where change is minimal
    # True = Keep Old Pixel (Smear), False = Update Pixel
    mask = diff < threshold
    
    # Stack mask for 3 channels
    mask3 = np.stack([mask]*3, axis=2)
    
    # Apply logic: Where mask is True, use Previous. Else use Current.
    return np.where(mask3, previous_arr, current_arr)

def crush_resolution(img):
    """Simulates bitrate death by downscaling and upscaling aggressively."""
    w, h = img.size
    small = img.resize((w//4, h//4), resample=Image.NEAREST)
    return small.resize((w, h), resample=Image.NEAREST)

# --- MAIN LOGIC ---

def main():
    v1 = "input.mp4"
    v2 = "image2.mp4"
    out_file = "output_abrasive.mp4"
    
    # Folders
    t_out = "frames_out"
    if os.path.exists(t_out): shutil.rmtree(t_out)
    os.makedirs(t_out, exist_ok=True)
    
    try:
        print(">>> EXTRACTING & DOWNSCALING (For Speed + Style)...")
        # Ensure files exist (LFS check)
        if not os.path.exists(v1) or os.path.getsize(v1) < 2000:
            raise FileNotFoundError("Input files are missing or empty LFS pointers.")

        # Audio Extraction
        has_audio = False
        try:
            run_ffmpeg(["ffmpeg", "-y", "-i", v1, "-vn", "-c:a", "aac", "temp.aac"])
            has_audio = True
        except: pass

        # Frame Extraction (To Memory-friendly temp files)
        # We extract directly to the small internal resolution
        w, h = SETTINGS['internal_res']
        run_ffmpeg(["ffmpeg", "-i", v1, "-vf", f"fps={SETTINGS['fps']},scale={w}:{h}", "f1_%05d.bmp"])
        run_ffmpeg(["ffmpeg", "-i", v2, "-vf", f"fps={SETTINGS['fps']},scale={w}:{h}", "f2_%05d.bmp"])
        
        frames1 = sorted([f for f in os.listdir(".") if f.startswith("f1_")])
        frames2 = sorted([f for f in os.listdir(".") if f.startswith("f2_")])
        
        total_frames = len(frames1) + len(frames2)
        print(f">>> PROCESSING {total_frames} FRAMES...")
        
        # State
        prev_arr = None
        stutter_buffer = None
        stutter_timer = 0
        
        for i in range(total_frames):
            print(f"MOSHING: {i}/{total_frames}", end='\r')
            
            # --- SOURCE SELECTION ---
            if i < len(frames1):
                # Load as Numpy Array immediately for speed
                curr_img = Image.open(frames1[i]).convert("RGB")
                
                # Transition Blend
                left = len(frames1) - i
                trans_len = int(len(frames1) * SETTINGS['transition_frac'])
                if left < trans_len and frames2:
                    # Dirty Difference Blend
                    idx2 = int((1 - (left/trans_len)) * (len(frames2)-1))
                    img2 = Image.open(frames2[idx2]).convert("RGB")
                    curr_img = ImageChops.difference(curr_img, img2)
            else:
                idx = i - len(frames1)
                if idx >= len(frames2): break
                curr_img = Image.open(frames2[idx]).convert("RGB")

            # Convert to Numpy
            curr_arr = np.array(curr_img)

            # --- STUTTER EFFECT (Freeze) ---
            if stutter_timer > 0:
                curr_arr = stutter_buffer
                stutter_timer -= 1
            elif random.random() < SETTINGS['stutter_chance']:
                stutter_buffer = curr_arr.copy()
                stutter_timer = random.randint(3, 10)

            # --- MOSH CORE (P-Frame Simulation) ---
            # This relies on prev_arr being the PREVIOUS PROCESSED FRAME
            moshed_arr = vector_mosh(curr_arr, prev_arr, SETTINGS['mosh_threshold'])
            
            # --- CHAOS INJECTION ---
            
            # 1. Pixel Sort (Melting)
            if random.random() < SETTINGS['sort_chance']:
                moshed_arr = vector_pixel_sort(moshed_arr)
                
            # 2. Invert (Flash)
            if random.random() < SETTINGS['invert_chance']:
                moshed_arr = 255 - moshed_arr
                
            # 3. Bloom / Bitrot (JPEG compression look)
            res = Image.fromarray(moshed_arr.astype('uint8'))
            if random.random() < SETTINGS['bloom_chance']:
                res = crush_resolution(res)
                # Update numpy array to match bloom
                moshed_arr = np.array(res)

            # Save and update state
            res.save(os.path.join(t_out, f"frame_{i:05d}.bmp")) # BMP is faster than PNG to write
            prev_arr = moshed_arr # CRITICAL: Feedback loop for smearing

        # Cleanup raw frames
        for f in frames1 + frames2: os.remove(f)

        print("\n>>> RENDERING FINAL VIDEO...")
        # Re-encode, Upscale to 720p with Nearest Neighbor (keeps blocks sharp)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(SETTINGS['fps']),
            "-i", f"{t_out}/frame_%05d.bmp",
            "-vf", "scale=1280:720:flags=neighbor", # Upscale sharp
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p", "temp_vid.mp4"
        ]
        run_ffmpeg(cmd)

        if has_audio:
            print(">>> DESTROYING AUDIO...")
            run_ffmpeg([
                "ffmpeg", "-y", "-i", "temp_vid.mp4", "-i", "temp.aac",
                "-af", get_audio_filter(),
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", out_file
            ])
        else:
            os.rename("temp_vid.mp4", out_file)

        print(f"DONE: {out_file}")

    except Exception as e:
        print(f"CRITICAL FAIL: {e}")
        exit(1)
    finally:
        if os.path.exists(t_out): shutil.rmtree(t_out)
        if os.path.exists("temp.aac"): os.remove("temp.aac")
        if os.path.exists("temp_vid.mp4"): os.remove("temp_vid.mp4")

if __name__ == "__main__":
    main()
